import argparse
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from rl4llm.data import load_multiple_datasets
from rl4llm.utils import save_to_json_file, save_to_parquet_file, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Build coherent dataset for training classifier')
    parser.add_argument(
        '--model-names',
        type=str,
        nargs='+',
        default=['Qwen/Qwen2.5-1.5B-Instruct'],
        help='List of LLM model names (default: [Qwen/Qwen2.5-1.5B-Instruct])',
    )
    parser.add_argument(
        '--target-model-name', type=str, default='Qwen/Qwen2.5-7B-Instruct', help='The target model we want to fine-tune'
    )
    parser.add_argument('--batch-size', type=int, default=10, help='Generation batch size (default: 10)')
    parser.add_argument(
        '--task-size', type=int, default=1000, help='Maximum number of samples per tasks to process (default: 1000)'
    )
    parser.add_argument('--max-tokens', type=int, default=4000, help='Maximum number of tokens per sample (default: 4000)')
    parser.add_argument('--split-ratio', type=float, default=0.9, help='Train/test split ratio (default: 0.9)')
    parser.add_argument('--seed', type=int, default=153, help='Runtime seed (default: 153)')
    parser.add_argument(
        '--save-dir',
        type=str,
        default='data/coherent_dataset',
        help='Directory to save the output files (default: data/coherent_dataset)',
    )

    return parser.parse_args()


@torch.inference_mode()
def generate_positive_samples(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: List[Dict],
    system_prompt: str,
    batch_size: int = 8,
    max_tokens: int = 4000,
) -> List[Dict]:
    """
    Generate completions for a list of prompts using an LLM with proper batching.

    Args:
        model: The pre-trained language model
        tokenizer: The tokenizer corresponding to the model
        dataset: List of dictionaries, each containing at least a "prompt" key
        system_prompt: System prompt to include with each generation
        batch_size: Number of samples to process at once
        max_tokens: Max generation token size

    Returns:
        List of dictionaries with prompt, completion, and completion_tokens
    """
    results = []
    model_name = model.config.name_or_path
    total_steps = len(dataset) // batch_size
    pbar = tqdm(desc='Generating samples', total=total_steps)
    # Process the dataset in batches
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]

        # Format each prompt as chat-style message with system prompt
        batch_messages = []
        for sample in batch:
            if system_prompt:
                messages = [
                    {'role': 'system', 'content': system_prompt.strip()},
                    {'role': 'user', 'content': sample['prompt'].strip()},
                ]
            else:
                messages = [{'role': 'user', 'content': sample['prompt'].strip()}]
            batch_messages.append(messages)

        formatted_prompt = tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)
        batch_inputs = tokenizer(
            formatted_prompt,
            return_tensors='pt',
            truncation=True,
            padding=True,
            padding_side='left',
        ).to(model.device)

        prompt_length = batch_inputs['input_ids'].size(1)

        outputs = model.generate(
            **batch_inputs,
            min_new_tokens=50,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Process each output in the batch
        for j, sample in enumerate(batch):
            output_ids = outputs.sequences[j]
            completion_ids = output_ids[prompt_length:]
            # Remove special tokens (EOS and PAD) from the completion using a mask
            mask = (completion_ids != tokenizer.eos_token_id) & (completion_ids != tokenizer.pad_token_id)
            completion_ids = completion_ids[mask]
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

            results.append(
                {
                    'completion': completion,
                    'completion_tokens': completion_ids.tolist(),
                    'label': 1,
                    'source': f"{sample['source']}_{model_name}" if 'source' in sample else model_name,
                }
            )

        pbar.update(1)

    pbar.close()
    return results


def generate_negative_samples(
    source_dataset: List[Dict],
    tokenizer: PreTrainedTokenizer,
    position: List[str] = ['beginning', 'middle', 'end', 'random', 'full'],
    levels: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7],
    max_tokens: int = 4000,
    repetition_probs: float = 0.2,
) -> List[Dict]:
    """
    Generate negative samples by manipulating tokens to create incoherent text.

    Args:
        source_dataset: List of dictionaries with prompt, prompt_tokens, completion, completion_tokens
        tokenizer: The tokenizer to use for encoding/decoding
        position: List of positions to apply manipulations (beginning, middle, end, random, full)
        levels: List of float between 0 and 1 indicating severity of manipulation (higher = more incoherent)
        max_tokens: Maximum number of tokens to process
        repetition_probs: Probability of adding repetition to the completion

    Returns:
        List of dictionaries with manipulated prompt and completion
    """

    assert repetition_probs > 0.0 and repetition_probs <= 1.0

    noise_tokens = precompute_noise_tokens(tokenizer)

    negative_samples = []
    vocab_size = tokenizer.vocab_size
    vocab_indices = list(range(vocab_size))

    for sample in source_dataset:
        completion_tokens = sample['completion_tokens']
        source = sample['source']
        method = 'random'

        # With 20% probability, add repetition
        if random.random() < repetition_probs and len(completion_tokens) > 50:
            # manipulated_completion_tokens = add_repetition(prompt_tokens[:max_tokens])
            manipulated_completion_tokens = add_repetition(completion_tokens[:max_tokens])
            method = 'repetition'
        else:
            # Apply token manipulations based on position and level
            level = random.choice(levels)
            selected_position = random.choice(position)
            manipulated_completion_tokens = manipulate_tokens(
                completion_tokens[:max_tokens], selected_position, level, vocab_indices, noise_tokens, tokenizer
            )

        # Convert tokens back to text
        manipulated_completion_text = tokenizer.decode(manipulated_completion_tokens)
        negative_sample = {
            'completion': manipulated_completion_text,
            'completion_tokens': manipulated_completion_tokens,
            'label': 0,
            'source': source,
            'method': method,
        }

        negative_samples.append(negative_sample)

    return negative_samples


def precompute_noise_tokens(tokenizer):
    """Precompute noise tokens for efficiency."""
    noise_chars = (
        '!@#$%^&*()_+-=[]{}|;:,.<>?~1234567890'
        'qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM'
        '¡¿¢£¥§©®¶...!!##@@--xXxzzz111typoerrwtflol'
    )
    noise_elements = list(noise_chars) + noise_chars.split()
    noise_tokens = []

    for elem in noise_elements:
        encoded = tokenizer.encode(elem, add_special_tokens=False)
        if encoded:  # Only add if encoding produced something
            if len(encoded) == 1:
                noise_tokens.append(encoded[0])
            else:
                noise_tokens.append(encoded)

    return noise_tokens


def manipulate_tokens(
    tokens: List[int],
    position: str,
    level: float,
    vocab_indices: List[int],
    noise_tokens: List[int],
    tokenizer: PreTrainedTokenizer,
) -> List[int]:
    """
    Apply token manipulations based on position and level.

    Args:
        tokens: List of token IDs
        position: Where to apply manipulations ('beginning', 'middle', 'end', 'random', 'full')
        level: Severity of manipulation (0-1)
        vocab_indices: List of valid token indices from tokenizer
        noise_tokens: List of noise tokens
        tokenizer: The tokenizer

    Returns:
        Manipulated token list
    """

    tokens = tokens.copy()  # Create a copy to avoid modifying the original

    if not tokens:
        return tokens

    # Calculate number of tokens to manipulate based on level
    num_tokens = len(tokens)
    num_to_manipulate = int(num_tokens * level)

    # Determine which indices to manipulate based on position
    indices_to_manipulate = []

    if position == 'beginning':
        start_idx = 0
        end_idx = min(num_to_manipulate, num_tokens)
        indices_to_manipulate = list(range(start_idx, end_idx))

    elif position == 'middle':
        start_idx = (num_tokens - num_to_manipulate) // 2
        end_idx = start_idx + num_to_manipulate
        indices_to_manipulate = list(range(start_idx, min(end_idx, num_tokens)))

    elif position == 'end':
        start_idx = max(0, num_tokens - num_to_manipulate)
        indices_to_manipulate = list(range(start_idx, num_tokens))

    elif position == 'random':
        indices_to_manipulate = random.sample(range(num_tokens), min(num_to_manipulate, num_tokens))

    elif position == 'full':
        indices_to_manipulate = list(range(num_tokens))
        # For "full", adjust the number based on level
        indices_to_manipulate = random.sample(indices_to_manipulate, min(num_to_manipulate, num_tokens))

    # Apply different manipulation techniques
    for idx in indices_to_manipulate:
        manipulation_type = random.choice(['swap', 'randomize', 'inject_noise'])

        if manipulation_type == 'swap' and idx < num_tokens - 1:
            # Swap adjacent tokens
            tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]

        elif manipulation_type == 'randomize':
            # Replace with another random token from the dataset
            tokens[idx] = random.choice(tokens)

        elif manipulation_type == 'inject_noise':
            # Insert random token from vocabulary
            tokens[idx] = random.choice(vocab_indices)

    # Apply sentence-level manipulations if level is high
    if level > 0.5 and len(tokens) > 50:
        # Extract sentence boundaries using punctuation tokens
        punct_ids = [tokenizer.encode('.')[0], tokenizer.encode('!')[0], tokenizer.encode('?')[0], tokenizer.encode(';')[0]]

        sent_breaks = [i for i, t in enumerate(tokens) if t in punct_ids]

        if len(sent_breaks) > 1:
            # Shuffle sentence order
            sentences = []
            prev_break = 0

            for brk in sent_breaks:
                sentences.append(tokens[prev_break : brk + 1])
                prev_break = brk + 1

            if prev_break < len(tokens):
                sentences.append(tokens[prev_break:])

            random.shuffle(sentences)
            tokens = [t for sentence in sentences for t in sentence]

    return tokens


def add_repetition(tokens: List[int], min_repeat: int = 4, max_repeat: int = 20, max_tokens: int = 4000) -> List[int]:
    """
    Add repetition to token sequences.

    Args:
        tokens: List of token IDs.
        min_repeat: Minimum number of times to repeat the segment.
        max_repeat: Maximum number of times to repeat the segment.

    Returns:
        Token list with repetitions inserted at a random location.
    """
    # Only apply repetition if the token list is sufficiently long.
    if len(tokens) < 20:
        return tokens

    # Determine a safe segment length: at least 5 tokens and at most the smaller of 100 or half the sequence.
    max_possible_seg = min(100, len(tokens) // 2)
    segment_len = random.randint(5, max_possible_seg)

    # Randomly choose a segment within tokens
    start_idx = random.randint(0, len(tokens) - segment_len)
    segment = tokens[start_idx : start_idx + segment_len]

    # Determine how many times to repeat the segment
    repeat_count = random.randint(min_repeat, max_repeat)

    # Choose a random insertion index (not necessarily at the end)
    insert_idx = random.randint(0, len(tokens))

    # Build the new token sequence with the repeated segment inserted
    repeated_segment = segment * repeat_count
    new_tokens = tokens[:insert_idx] + repeated_segment + tokens[insert_idx:]

    return new_tokens[:max_tokens]


def extract_completion_from_item(item: Dict) -> str:
    """Try to extract the completion from different possible field names/structures."""
    completion = None

    # Extract the completion from different possible field names/structures
    if 'completion' in item and isinstance(item['completion'], str):
        completion = item['completion']
    elif 'generations' in item and isinstance(item['generations'], list) and len(item['generations']) >= 1:  # from OpenR1 Math
        completion = random.choice(item['generations'])
    elif 'messages' in item and isinstance(item['messages'], list):  # standard SFT chat
        last_turn = item['messages'][-1]
        if 'role' not in last_turn or 'content' not in last_turn or last_turn['role'] == 'assistant':
            return None
        else:
            completion = last_turn['content']

    return completion


def convert_dataset_to_positive_samples(dataset: List[Dict], tokenizer: PreTrainedTokenizer, max_tokens: int) -> List[Dict]:
    """
    Convert existing dataset to positive samples format.

    Args:
        dataset: List of dictionaries with question/problem and completion/generations
        tokenizer: The tokenizer to encode the text
        max_tokens: Maximum number of tokens to include

    Returns:
        List of dictionaries with prompt, prompt_tokens, completion, completion_tokens
    """
    samples = []

    for item in dataset:
        completion = extract_completion_from_item(item)
        if not completion:
            continue

        # Tokenize prompt and completion
        completion_tokens = tokenizer.encode(completion, max_length=max_tokens, add_special_tokens=False)[:max_tokens]

        # Create the sample
        samples.append(
            {
                'completion': completion,
                'completion_tokens': completion_tokens,
                'label': 1,
                'source': item['source'] if 'source' in item else 'unknown',
            }
        )

    return samples


def generate_samples_with_local_llms(
    tokenizer: PreTrainedTokenizer,
    device: torch.device,
    torch_dtype: torch.dtype,
    args: argparse.Namespace,
    logger,
) -> Dict[str, List[Dict]]:
    """Run local LLM to generate samples."""

    system_prompt = """
    A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
    The Assistant first thinks about the reasoning process internally and then provides the user with the answer.
    The reasoning process and answer are enclosed within `<think> </think>` and `<answer> </answer>` tags, respectively. i.e.,
    `<think> Unstructured, free-form reasoning process exploring the problem in depth </think>`
    `<answer> A clear and concise high-level summary of the solution and final answer </answer>`
    """

    dual_lang_system_prompt = f"""{system_prompt}


The assistant first provide the reasoning process and answer in English, then translate them in Chinese.
"""

    gen_positive_samples = []
    gen_negative_samples = []

    train_ds, test_ds = load_multiple_datasets(['GSM', 'MATH'])
    full_ds = list(train_ds) + list(test_ds)
    full_df = pd.DataFrame(full_ds)
    full_df.rename(columns={'question': 'prompt', 'task_type': 'source'}, inplace=True)

    # use multiple models to generate diverse samples
    for model_name in args.model_names:
        logger.info(f"Loading tokenizer {model_name}")
        # only use this tokenizer for generation
        local_tokenizer = AutoTokenizer.from_pretrained(model_name)
        model_args = {
            'pretrained_model_name_or_path': model_name,
            'torch_dtype': torch_dtype,
            'use_cache': True,
            'attn_implementation': 'flash_attention_2',
            'pad_token_id': tokenizer.pad_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }
        model = AutoModelForCausalLM.from_pretrained(**model_args)
        model = model.to(device)

        selected_df = filter_ds_by_group_size(
            full_df, group_by='source', group_size=int(args.task_size), seed=random.choice(range(1, 9999))
        )
        selected_ds = list(selected_df.to_dict(orient='records'))
        random.shuffle(selected_ds)

        logger.info(f'Start to generate {len(selected_ds)} positive samples...')
        llm_positive_samples = generate_positive_samples(
            model, local_tokenizer, selected_ds, system_prompt, args.batch_size, max_tokens=args.max_tokens
        )

        # generate another small subset of positive samples with multiple languages
        sub_df = filter_ds_by_group_size(
            full_df, group_by='source', group_size=int(args.task_size * 0.2), seed=random.choice(range(1, 9999))
        )
        sub_ds = list(sub_df.to_dict(orient='records'))
        random.shuffle(sub_ds)

        logger.info(f'Start to generate {len(sub_ds)} positive samples with dual language...')
        llm_positive_samples_dual_lang = generate_positive_samples(
            model, local_tokenizer, sub_ds, dual_lang_system_prompt, args.batch_size, max_tokens=args.max_tokens
        )

        llm_positive_samples = llm_positive_samples + llm_positive_samples_dual_lang

        # always use the same tokenizer for negative samples
        llm_negative_samples = generate_negative_samples(llm_positive_samples, tokenizer, max_tokens=args.max_tokens)

        gen_positive_samples.extend(llm_positive_samples)
        gen_negative_samples.extend(llm_negative_samples)

        torch.cuda.empty_cache()

    return {'positive': gen_positive_samples, 'negative': gen_negative_samples}


def generate_strong_cot_samples(tokenizer: PreTrainedTokenizer, args: argparse.Namespace) -> Dict[str, List[Dict]]:
    """Generates strong CoT samples"""
    cot_dataset = load_dataset('open-r1/OpenR1-Math-220k', 'default')
    cot_ds_df = pd.DataFrame(list(cot_dataset['train']))
    # For each source, take at most N items.
    cot_ds_df = filter_ds_by_group_size(
        cot_ds_df, group_by='source', group_size=args.task_size, seed=random.choice(range(1, 9999))
    )
    cot_dataset = list(cot_ds_df.to_dict(orient='records'))
    cot_positive_samples = convert_dataset_to_positive_samples(cot_dataset, tokenizer, max_tokens=args.max_tokens)
    cot_negative_samples = generate_negative_samples(cot_positive_samples, tokenizer, max_tokens=args.max_tokens)

    return {'positive': cot_positive_samples, 'negative': cot_negative_samples}


def generate_mixed_sft_samples(tokenizer: PreTrainedTokenizer, args: argparse.Namespace) -> Dict[str, List[Dict]]:
    """Generates SFT samples"""

    # full list of sources from "allenai/tulu-3-sft-mixture"
    mixed_dataset = load_dataset('allenai/tulu-3-sft-mixture')
    # sources_to_use = [
    #     'ai2-adapt-dev/personahub_math_v5_regen_149960',
    #     'ai2-adapt-dev/personahub_ifdata_manual_seed_v3_29980',
    #     'ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k',
    #     'ai2-adapt-dev/no_robots_converted',
    #     'ai2-adapt-dev/numinamath_tir_math_decontaminated',
    #     'ai2-adapt-dev/tulu_v3.9_wildchat_100k',
    #     'ai2-adapt-dev/flan_v2_converted',
    #     'ai2-adapt-dev/tulu_hard_coded_repeated_10',
    #     'ai2-adapt-dev/tulu_v3.9_aya_100k',
    #     'ai2-adapt-dev/oasst1_converted',
    #     'ai2-adapt-dev/tulu_v3.9_wildjailbreak_decontaminated_50k',
    #     'ai2-adapt-dev/tulu_v3.9_table_gpt_5k',
    #     'ai2-adapt-dev/personahub_code_v2_34999',
    #     'ai2-adapt-dev/evol_codealpaca_heval_decontaminated',
    #     'allenai/tulu-3-sft-personas-math-grade',
    #     'ai2-adapt-dev/tulu_v3.9_synthetic_finalresp_wildguardmixtrain_decontaminated_50k',
    #     'ai2-adapt-dev/tulu_v3.9_sciriff_10k',
    #     'ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k',
    #     'ai2-adapt-dev/coconot_converted',
    # ]
    sources_to_use = [
        'ai2-adapt-dev/tulu_v3.9_personahub_math_interm_algebra_20k',
        'ai2-adapt-dev/numinamath_tir_math_decontaminated',
        'ai2-adapt-dev/tulu_v3.9_wildchat_100k',
        'ai2-adapt-dev/tulu_hard_coded_repeated_10',
        'ai2-adapt-dev/tulu_v3.9_aya_100k',
        'ai2-adapt-dev/evol_codealpaca_heval_decontaminated',
        'ai2-adapt-dev/tulu_v3.9_open_math_2_gsm8k_50k',
        'ai2-adapt-dev/tulu_v3.9_sciriff_10k',
    ]
    # filter the dataset with only the sources we want
    mixed_ds_df = pd.DataFrame(list(mixed_dataset['train']))
    mixed_ds_df = mixed_ds_df[mixed_ds_df['source'].isin(sources_to_use)]

    # Filter that the last item in 'messages' has role of assistant, and the content has length of greater than 200,
    def last_message_valid(row):
        messages = row.get('messages', [])
        # Ensure messages is a non-empty list
        if isinstance(messages, list) and messages:
            last_message = messages[-1]
            # Check if last message has the required role and content length
            return last_message.get('role') == 'assistant' and len(last_message.get('content', '')) > 200
        return False

    mixed_ds_df = mixed_ds_df[mixed_ds_df.apply(last_message_valid, axis=1)]

    # For each source, take at most N items for each source
    mixed_ds_df = filter_ds_by_group_size(
        mixed_ds_df, group_by='source', group_size=args.task_size, seed=random.choice(range(1, 9999))
    )
    mixed_dataset = list(mixed_ds_df.to_dict(orient='records'))

    mixed_positive_samples = convert_dataset_to_positive_samples(mixed_dataset, tokenizer, max_tokens=args.max_tokens)
    mixed_negative_samples = generate_negative_samples(mixed_positive_samples, tokenizer, max_tokens=args.max_tokens)

    return {'positive': mixed_positive_samples, 'negative': mixed_negative_samples}


def filter_ds_by_group_size(ds: pd.DataFrame, group_by: str, group_size: int, seed: int = 42) -> pd.DataFrame:
    """Filter a dataset by group size."""
    return ds.groupby(group_by, group_keys=False).apply(lambda x: x.sample(n=min(len(x), group_size), random_state=seed))


def main():
    """Main function to build coherent dataset."""
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    logger = setup_logger()

    torch_dtype = torch.float16
    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        if torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
    else:
        device = torch.device('cpu')

    # all negative samples will be generated using the target model tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.target_model_name)
    sample_list = []

    logger.info('Run local LLMs to generate samples...')
    llm_gen_result = generate_samples_with_local_llms(
        tokenizer=tokenizer,
        device=device,
        torch_dtype=torch_dtype,
        args=args,
        logger=logger,
    )
    sample_list.append(llm_gen_result)

    logger.info('Loading DeepSeek-R1 Math CoT dataset and generate samples...')
    cot_result = generate_strong_cot_samples(tokenizer=tokenizer, args=args)
    sample_list.append(cot_result)

    logger.info('Loading mixed SFT dataset and generate samples...')
    mixed_sft_results = generate_mixed_sft_samples(tokenizer=tokenizer, args=args)
    sample_list.append(mixed_sft_results)

    # now split into train and test by ratio from each task
    train_samples = []
    test_samples = []
    keys_to_keep = ('completion', 'label', 'source')
    for sample_dict in sample_list:
        pos_samples = sample_dict['positive']
        neg_samples = sample_dict['negative']
        random.shuffle(pos_samples)
        random.shuffle(neg_samples)

        pos_split_index = int(len(pos_samples) * args.split_ratio)
        neg_split_index = int(len(neg_samples) * args.split_ratio)
        train_samples.extend(pos_samples[:pos_split_index])
        test_samples.extend(pos_samples[pos_split_index:])

        # only keep required fields
        train_samples.extend([{key: item[key] for key in keys_to_keep} for item in neg_samples[:neg_split_index]])
        test_samples.extend([{key: item[key] for key in keys_to_keep} for item in neg_samples[neg_split_index:]])

    random.shuffle(train_samples)
    random.shuffle(test_samples)

    # compute stats
    stats = {
        'train_size': len(train_samples),
        'test_size': len(test_samples),
        'positive_train_size': len([item for item in train_samples if item['label'] == 1]),
        'positive_test_size': len([item for item in test_samples if item['label'] == 1]),
        'negative_train_size': len([item for item in train_samples if item['label'] == 0]),
        'negative_test_size': len([item for item in test_samples if item['label'] == 0]),
    }

    # Save files
    logger.info(f"Saving coherent dataset to {args.save_dir}")
    save_to_parquet_file(train_samples, f"{args.save_dir}/train.parquet", compression='snappy')
    save_to_parquet_file(test_samples, f"{args.save_dir}/test.parquet", compression='snappy')
    save_to_json_file(stats, f"{args.save_dir}/metadata.json")
    logger.info('Dataset creation completed successfully')


if __name__ == '__main__':
    main()
