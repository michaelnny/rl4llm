import argparse
import os
import random
import re
import time
from multiprocessing import Pool, cpu_count
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from rl4llm.data import load_multiple_datasets
from rl4llm.utils import load_from_jsonl_file, save_to_json_file, save_to_parquet_file, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description='Build coherent dataset for training classifier')
    parser.add_argument(
        '--load-custom-source',
        type=str,
        default='data/cot_data/mixed_gsm_math_positive_samples_gpt4.jsonl.gz',
        help='Directory to load custom positive samples',
    )
    parser.add_argument('--max-size', type=int, default=10000, help='Maximum number of samples to process (default: 10000)')
    parser.add_argument('--max-tokens', type=int, default=4000, help='Maximum number of tokens per sample (default: 4000)')
    parser.add_argument('--split-ratio', type=float, default=0.9, help='Train/test split ratio (default: 0.9)')
    parser.add_argument('--seed', type=int, default=153, help='Runtime seed (default: 153)')
    parser.add_argument(
        '--save-dir',
        type=str,
        default='data/coherent_dataset',
        help='Directory to save the output files (default: data/coherent_dataset)',
    )
    parser.add_argument(
        '--model-name',
        type=str,
        default='Qwen/Qwen2.5-3B-Instruct',
        help='Tokenizer model name (default: Qwen/Qwen2.5-3B-Instruct)',
    )
    return parser.parse_args()


def generate_text_repetition(args) -> str:
    """
    Generate repetitive text from input, without introducing new content.
    Only repeats existing content with minimal modifications.

    Returns:
        The generated repetitive text
    """
    input_text, repetition_count_min, repetition_count_max = args
    assert repetition_count_max > repetition_count_min and repetition_count_min > 3

    repetition_level = random.choice(['sentence', 'block'])
    variation_position = random.choice(['beginning', 'middle', 'end'])

    # Sentence splitting pattern - handles periods, question marks, exclamation points
    sentence_pattern = re.compile(r'(?<=[.!?])\s+')

    # Paragraph splitting pattern
    block_pattern = re.compile(r'\n\s*\n')

    # Split text based on repetition level
    if repetition_level == 'sentence':
        # Split by sentences
        units = sentence_pattern.split(input_text)
        # Add back the period that was removed during splitting
        units = [
            unit + '.' if not unit.endswith(('.', '!', '?')) and i < len(units) - 1 else unit for i, unit in enumerate(units)
        ]
    else:  # block level
        # Split by paragraphs/blocks
        units = block_pattern.split(input_text)

    if len(units) <= 1:
        # Not enough content to create meaningful repetition
        return input_text

    # Determine which positions to repeat
    positions_to_repeat = []

    if variation_position == 'beginning':
        segment_size = min(3, len(units) // 3)
        positions_to_repeat = list(range(segment_size))
    elif variation_position == 'middle':
        start = len(units) // 3
        end = 2 * len(units) // 3
        segment_size = min(3, (end - start))
        positions_to_repeat = list(range(start, start + segment_size))
    elif variation_position == 'end':
        segment_size = min(3, len(units) // 3)
        positions_to_repeat = list(range(len(units) - segment_size, len(units)))
    else:  # random
        # Choose a random contiguous segment
        segment_size = min(3, max(1, len(units) // 4))
        start = random.randint(0, max(0, len(units) - segment_size))
        positions_to_repeat = list(range(start, start + segment_size))

    # Build the result with repetitions
    result = []
    for i, unit in enumerate(units):
        result.append(unit)

        # If this unit is marked for repetition, repeat it
        if i in positions_to_repeat:
            # Random number of repetitions within the specified range
            repetition_count = random.randint(repetition_count_min, repetition_count_max)

            for _ in range(repetition_count):
                # Just repeat the unit without modifications
                result.append(unit)

    # Join the results based on the repetition level
    if repetition_level == 'sentence':
        # Join with spaces
        joined_result = ' '.join(result)
        # Clean up any double spaces
        joined_result = re.sub(r'\s+', ' ', joined_result)
        # Clean up any double periods
        joined_result = re.sub(r'\.\.', '.', joined_result)
    else:  # block level
        # Join with double newlines
        joined_result = '\n\n'.join(result)

    return joined_result


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

    # Process the dataset in batches
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]

        # Format each prompt as chat-style message with system prompt
        batch_messages = []
        for sample in batch:
            if system_prompt:
                messages = [{'role': 'system', 'content': system_prompt}, {'role': 'user', 'content': sample['prompt']}]
            else:
                messages = [{'role': 'user', 'content': sample['prompt']}]
            batch_messages.append(messages)

        # Convert all messages to model input format in a single batch
        batch_inputs = [tokenizer.apply_chat_template(messages, return_tensors='pt') for messages in batch_messages]

        # Record input lengths for extracting completions later
        input_lengths = [len(inputs) for inputs in batch_inputs]

        # Pad inputs to the same length for batched inference
        padded_inputs = tokenizer.pad({'input_ids': [inputs for inputs in batch_inputs]}, padding=True, return_tensors='pt')

        # Move inputs to the same device as the model
        padded_inputs = {k: v.to(model.device) for k, v in padded_inputs.items()}

        # Generate outputs for the entire batch at once
        outputs = model.generate(
            **padded_inputs,
            min_new_tokens=50,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Process each item in the batch
        for j, (sample, input_length) in enumerate(zip(batch, input_lengths)):
            # Extract completion sequence
            output_ids = outputs.sequences[j]
            completion_ids = output_ids[input_length:]
            prompt_ids = batch_inputs[j]

            # Remove special tokens from completion
            completion_ids = [
                token_id for token_id in completion_ids if token_id not in [tokenizer.eos_token_id, tokenizer.pad_token_id]
            ]

            # Decode the completion
            completion = tokenizer.decode(completion_ids, skip_special_tokens=True)

            # Add to results
            results.append(
                {
                    'prompt': sample['prompt'],
                    'prompt_tokens': prompt_ids.tolist() if isinstance(prompt_ids, torch.Tensor) else prompt_ids,
                    'completion': completion,
                    'completion_tokens': (
                        completion_ids.tolist() if isinstance(completion_ids, torch.Tensor) else completion_ids
                    ),
                    'label': 1,
                }
            )

        # Optional: Print progress
        print(f"Processed {min(i + batch_size, len(dataset))}/{len(dataset)} samples")

    return results


def generate_negative_samples(
    source_dataset: List[Dict],
    tokenizer: PreTrainedTokenizer,
    position: List[str] = ['beginning', 'middle', 'end', 'random', 'full'],
    levels: List[float] = [0.4, 0.5, 0.6, 0.7],
    max_tokens: int = 4000,
) -> List[Dict]:
    """
    Generate negative samples by manipulating tokens to create incoherent text.

    Args:
        source_dataset: List of dictionaries with prompt, prompt_tokens, completion, completion_tokens
        tokenizer: The tokenizer to use for encoding/decoding
        position: List of positions to apply manipulations (beginning, middle, end, random, full)
        levels: List of float between 0 and 1 indicating severity of manipulation (higher = more incoherent)
        max_tokens: Maximum number of tokens to process

    Returns:
        List of dictionaries with manipulated prompt and completion
    """

    noise_tokens = precompute_noise_tokens(tokenizer)

    negative_samples = []
    vocab_size = tokenizer.vocab_size
    vocab_indices = list(range(vocab_size))

    # Select a random position for each sample if multiple positions are provided
    for sample in source_dataset:
        prompt_tokens = sample['prompt_tokens']
        completion_tokens = sample['completion_tokens']

        level = random.choice(levels)

        # With 20% probability, add repetition
        if random.random() < 0.2 and len(prompt_tokens) > 50 and len(completion_tokens) > 50:
            manipulated_completion_tokens = add_repetition(prompt_tokens[:max_tokens])
            manipulated_completion_tokens = add_repetition(completion_tokens[:max_tokens])
        else:
            # Apply token manipulations based on position and level
            selected_position = random.choice(position)
            manipulated_prompt_tokens = manipulate_tokens(
                prompt_tokens[:max_tokens], selected_position, level, vocab_indices, noise_tokens, tokenizer
            )

            selected_position = random.choice(position)
            manipulated_completion_tokens = manipulate_tokens(
                completion_tokens[:max_tokens], selected_position, level, vocab_indices, noise_tokens, tokenizer
            )

        # Convert tokens back to text
        manipulated_prompt_text = tokenizer.decode(manipulated_prompt_tokens)
        manipulated_completion_text = tokenizer.decode(manipulated_completion_tokens)

        # Create the negative sample
        negative_sample = {
            'prompt': manipulated_prompt_text,
            'prompt_tokens': manipulated_prompt_tokens,
            'completion': manipulated_completion_text,
            'completion_tokens': manipulated_completion_tokens,
            'label': 0,
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
        manipulation_type = random.choice(['swap', 'randomize', 'inject_noise', 'corrupt'])

        if manipulation_type == 'swap' and idx < num_tokens - 1:
            # Swap adjacent tokens
            tokens[idx], tokens[idx + 1] = tokens[idx + 1], tokens[idx]

        elif manipulation_type == 'randomize':
            # Replace with another random token from the dataset
            tokens[idx] = random.choice(tokens)

        elif manipulation_type == 'inject_noise':
            # Insert random token from vocabulary
            tokens[idx] = random.choice(vocab_indices)

        elif manipulation_type == 'corrupt':
            # Corrupt by using out-of-distribution characters/tokens
            # This creates more garbage-looking text at higher levels
            if level > 0.7:
                tokens[idx] = random.choice(noise_tokens)
            else:
                # Mildly corrupted - use valid but unrelated tokens
                tokens[idx] = random.choice(vocab_indices)

    # Apply sentence-level manipulations if level is high
    if level > 0.5 and len(tokens) > 20:
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


def add_repetition(tokens: List[int], min: int = 4, max: int = 20) -> List[int]:
    """
    Add repetition to token sequences.

    Args:
        tokens: List of token IDs

    Returns:
        Token list with repetitions
    """

    assert min >= 4

    if len(tokens) < 20:
        return tokens

    # Select a segment to repeat
    segment_len = random.randint(3, min(15, 100))
    start_idx = random.randint(0, len(tokens) - segment_len)
    segment = tokens[start_idx : start_idx + segment_len]

    # Repeat it at least 4 times
    repeat_count = random.randint(min, max)

    # Choose where to insert the repetition
    insert_idx = random.randint(0, len(tokens))

    # Create the new token sequence with repetition
    result = tokens[:insert_idx]
    for _ in range(repeat_count):
        result.extend(segment)
    result.extend(tokens[insert_idx:])

    return result


def convert_cot_data_to_positive_samples(dataset: List[Dict], tokenizer: PreTrainedTokenizer, max_tokens: int) -> List[Dict]:
    """
    Convert chain-of-thought dataset to positive samples format.

    Args:
        dataset: List of dictionaries with question/problem and completion/generations
        tokenizer: The tokenizer to encode the text
        max_tokens: Maximum number of tokens to include

    Returns:
        List of dictionaries with prompt, prompt_tokens, completion, completion_tokens
    """
    samples = []

    for item in dataset:
        prompt = None
        completion = None

        # Extract the prompt from different possible field names
        if 'question' in item:
            prompt = item['question']
        elif 'problem' in item:
            prompt = item['problem']

        # Extract the completion from different possible field names/structures
        if 'completion' in item and isinstance(item['completion'], str):
            completion = item['completion']
        elif (
            'generations' in item and isinstance(item['generations'], list) and len(item['generations']) >= 1
        ):  # from OpenR1 Math
            completion = random.choice(item['generations'])

        if not prompt or not completion:
            continue

        # Tokenize prompt and completion
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)[:max_tokens]
        completion_tokens = tokenizer.encode(completion, add_special_tokens=False)[:max_tokens]

        # Create the sample
        samples.append(
            {
                'prompt': prompt,
                'prompt_tokens': prompt_tokens,
                'completion': completion,
                'completion_tokens': completion_tokens,
            }
        )

    return samples


SYSTEM_PROMPT = """
A conversation between User and Assistant. The user asks a question, and the Assistant solves it.
The Assistant first thinks about the reasoning process internally and then provides the user with the answer.
The reasoning process and answer are enclosed within `<think> </think>` and `<answer> </answer>` tags, respectively. i.e.,
`<think> Unstructured, free-form reasoning process exploring the problem in depth </think>`
`<answer> A clear and concise high-level summary of the solution and final answer </answer>`
"""


def main():
    """Main function to build coherent dataset."""
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    logger = setup_logger()

    device = torch.device('cuda')
    torch_dtype = torch.bfloat16

    logger.info(f"Loading tokenizer {args.model_name}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model_args = {
        'pretrained_model_name_or_path': args.model_name,
        'torch_dtype': torch_dtype,
        'use_cache': False,
        'attn_implementation': 'flash_attention_2',
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token_id': tokenizer.eos_token_id,
    }
    model = AutoModelForCausalLM.from_pretrained(**model_args)
    model = model.to(device)

    logger.info('Loading dataset and use local LLM to generate samples...')
    loaded_train_dataset, loaded_test_dataset = load_multiple_datasets(['GSM', 'MATH'])
    loaded_dataset = list(loaded_train_dataset) + list(loaded_test_dataset)
    loaded_dataset = [{'prompt': item['question']} for item in loaded_dataset]
    random.shuffle(loaded_dataset)
    loaded_dataset = loaded_dataset[:200]

    positive_samples = generate_positive_samples(
        model, tokenizer, loaded_dataset, SYSTEM_PROMPT, 16, max_tokens=args.max_tokens
    )

    negative_samples = generate_negative_samples(positive_samples, tokenizer, max_tokens=args.max_tokens)

    # logger.info('Loading dataset OpenR1-Math-220k')
    # cot_dataset = load_dataset('open-r1/OpenR1-Math-220k', 'default')

    # # Apply max_size limit and shuffle
    # cot_dataset = cot_dataset['train'].shuffle().select(range(5000))
    # # Convert dataset to list if it's not already
    # cot_dataset = list(cot_dataset)

    # if args.load_custom_source:
    #     logger.info("Loading custom dataset...")
    #     try:
    #         custom_dataset = load_from_jsonl_file(args.load_custom_source)
    #         if len(custom_dataset) > args.max_size:
    #             random.shuffle(custom_dataset)
    #             custom_dataset = custom_dataset[: args.max_size]
    #         logger.info(f"Loaded custom {len(custom_dataset)} items")

    #     except Exception:
    #         pass

    # # Save files
    # logger.info(f"Saving coherent dataset to {args.save_dir}")
    # save_to_parquet_file(train_samples, f"{args.save_dir}/train.parquet", compression='snappy')
    # save_to_parquet_file(test_samples, f"{args.save_dir}/test.parquet", compression='snappy')
    # save_to_json_file(stats, f"{args.save_dir}/metadata.json")
    # logger.info('Dataset creation completed successfully')


if __name__ == '__main__':
    main()
