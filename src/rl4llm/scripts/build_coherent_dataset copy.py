import argparse
import os
import random
import re
import time
from multiprocessing import Pool, cpu_count

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

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
    # parser.add_argument('--max-size', type=int, default=10000, help='Maximum number of samples to process (default: 10000)')
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


def create_repetition_negative_samples(input_texts, repetition_count_min, repetition_count_max, num_workers=None):
    """
    Process a batch of texts in parallel.

    Args:
        input_texts: List of input texts
        repetition_count_min: Minimum number of repetitions.
        repetition_count_max: Maximum number of repetitions.
        num_workers: Maximum number of worker processes

    Returns:
        List of processed texts
    """
    if num_workers is None:
        num_workers = cpu_count()

    batch_args = [(text, repetition_count_min, repetition_count_max) for text in input_texts if len(text) > 200]
    with Pool(processes=num_workers) as pool:
        results = pool.map(generate_text_repetition, batch_args)

    return results


def extract_reason_traces(item, k=1):
    samples = []
    if 'generations' in item:  # from OpenR1 Math
        if len(item['generations']) >= k:
            samples = random.choices(item['generations'], k=k)
    elif 'completion' in item and isinstance(item['completion'], str):  # our custom dataset
        samples = [item['completion']]

    # add small amount of short samples
    if 'question' in item and isinstance(item['question'], str):
        if random.random() < 0.2:
            samples.append(item['question'])
    elif 'problem' in item and isinstance(item['problem'], str):
        if random.random() < 0.2:
            samples.append(item['problem'])

    return samples


def tokenize_and_chunk(text, tokenizer, max_tokens):
    """Tokenize text and handle chunking if needed."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) <= 100:
        return []

    if len(tokens) <= 1000:
        return [tokens]

    chunks = []
    start = 0
    min_size = int(0.3 * max_tokens)
    c = 0
    while start < len(tokens) and c < 2:
        # Random chunk size between 30% and 100% of max_tokens
        chunk_size = random.randint(min_size, max_tokens)
        chunk = tokens[start : start + chunk_size]
        if len(chunk) >= min_size:  # Only keep chunks above minimum size
            chunks.append(chunk)
            c += 1
        start += chunk_size
    return chunks


def process_positive_batch(args_tuple):
    """Process a batch of items to extract and tokenize positive samples."""
    batch_items, tokenizer, max_tokens = args_tuple
    positive_token_chunks = []

    for item in batch_items:
        # Make sure item is a dictionary, not a string
        if isinstance(item, dict):
            samples = extract_reason_traces(item, k=1)
            for sample in samples:
                token_chunks = tokenize_and_chunk(sample, tokenizer, max_tokens)
                positive_token_chunks.extend(token_chunks)

    return positive_token_chunks


def extract_positive_samples(dataset, tokenizer, max_tokens, num_workers=None, batch_size=100):
    """Extract and tokenize positive samples from the dataset in parallel."""
    if num_workers is None:
        num_workers = cpu_count()

    # Create batches of dataset items
    batches = [dataset[i : i + batch_size] for i in range(0, len(dataset), batch_size)]
    batch_args = [(batch, tokenizer, max_tokens) for batch in batches]

    # Process batches in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_positive_batch, batch_args)

    # Flatten the results and truncate to max_size
    positive_token_chunks = [chunk for result in results for chunk in result]

    return positive_token_chunks


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


def create_incoherent_from_tokens(
    tokens,
    location='random',
    intensity=0.5,
    operations=['swap', 'substitute', 'repeat', 'insert_noise'],
    vocab_size=None,
    noise_tokens=None,
):
    """Create incoherence directly from tokens without extra tokenization."""
    # Make a copy of the tokens
    total_tokens = len(tokens)
    if total_tokens <= 1:
        return tokens

    # Determine segment to transform
    segment_size = int(total_tokens * intensity)
    segment_size = max(1, segment_size)

    if location == 'beginning':
        start_idx, end_idx = 0, min(segment_size, total_tokens)
    elif location == 'end':
        start_idx, end_idx = max(0, total_tokens - segment_size), total_tokens
    elif location == 'middle':
        mid_point = total_tokens // 2
        half_segment = segment_size // 2
        start_idx, end_idx = max(0, mid_point - half_segment), min(total_tokens, mid_point + half_segment)
    elif location == 'random':
        if total_tokens <= segment_size:
            start_idx, end_idx = 0, total_tokens
        else:
            start_idx = random.randint(0, total_tokens - segment_size)
            end_idx = start_idx + segment_size
    else:  # "full"
        start_idx, end_idx = 0, total_tokens

    # Work on a NumPy array for faster manipulation
    modified_tokens = np.array(tokens, dtype=np.int32)

    # Define indices to modify
    indices_to_modify = np.arange(start_idx, end_idx)
    num_to_modify = max(1, int(total_tokens * intensity))
    num_to_modify = min(num_to_modify, len(indices_to_modify))
    selected_indices = np.random.choice(indices_to_modify, num_to_modify, replace=False)

    # Apply transformations
    for idx in selected_indices:
        if idx >= len(modified_tokens):
            continue
        operation = random.choice(operations)
        if operation == 'swap' and len(modified_tokens) > 1:
            swap_idx = random.randint(0, len(modified_tokens) - 1)
            while swap_idx == idx:
                swap_idx = random.randint(0, len(modified_tokens) - 1)
            modified_tokens[idx], modified_tokens[swap_idx] = modified_tokens[swap_idx], modified_tokens[idx]
        elif operation == 'substitute':
            modified_tokens[idx] = random.randint(0, vocab_size - 1)
        elif operation == 'repeat':
            repeat_count = random.randint(1, 3)
            modified_tokens = np.insert(modified_tokens, idx + 1, [modified_tokens[idx]] * repeat_count)
            selected_indices = np.where(selected_indices > idx, selected_indices + repeat_count, selected_indices)
        elif operation == 'insert_noise':
            noise_element = random.choice(noise_tokens)
            if isinstance(noise_element, int):
                modified_tokens = np.insert(modified_tokens, idx, noise_element)
                selected_indices = np.where(selected_indices >= idx, selected_indices + 1, selected_indices)
            else:
                modified_tokens = np.insert(modified_tokens, idx, noise_element)
                selected_indices = np.where(selected_indices >= idx, selected_indices + len(noise_element), selected_indices)

    return modified_tokens.tolist()


def process_negative_batch(args_tuple):
    """Process a batch of token samples to create incoherent versions."""
    token_samples, locations, intensities, vocab_size, noise_tokens, max_tokens = args_tuple
    batch_result = []

    for tokens in token_samples:
        location = random.choice(locations)
        intensity = random.choice(intensities)
        incoherent_tokens = create_incoherent_from_tokens(
            tokens, location=location, intensity=intensity, vocab_size=vocab_size, noise_tokens=noise_tokens
        )

        # Re-chunk if the token length exceeds the maximum
        if len(incoherent_tokens) > max_tokens:
            min_chunk_size = int(0.6 * max_tokens)
            chunks = [incoherent_tokens[i : i + max_tokens] for i in range(0, len(incoherent_tokens), max_tokens)]
            valid_chunks = [chunk for chunk in chunks if len(chunk) >= min_chunk_size]
            batch_result.extend(valid_chunks)
        else:
            batch_result.append(incoherent_tokens)

    return batch_result


def extract_negative_samples(
    positive_token_samples,
    num_workers=None,
    batch_size=1000,
    locations=None,
    intensities=None,
    vocab_size=None,
    noise_tokens=None,
    max_tokens=2048,
):
    """Create incoherent samples from tokenized positive samples using multiprocessing."""
    if locations is None:
        locations = ['beginning', 'middle', 'end', 'random', 'full']
    if intensities is None:
        intensities = [round(x, 2) for x in np.arange(0.4, 0.8, 0.05)]
    if num_workers is None:
        num_workers = cpu_count()

    # Create batches of tokenized samples
    batches = [positive_token_samples[i : i + batch_size] for i in range(0, len(positive_token_samples), batch_size)]
    batch_args = [(batch, locations, intensities, vocab_size, noise_tokens, max_tokens) for batch in batches]

    # Process batches in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(process_negative_batch, batch_args)

    # Flatten results
    return [item for sublist in results for item in sublist]


def collect_token_stats(positive_token_samples, negative_token_samples):
    """Collect token statistics for both positive and negative samples."""

    # Positive samples stats
    positive_lengths = [len(tokens) for tokens in positive_token_samples]

    # Negative samples stats
    negative_lengths = [len(tokens) for tokens in negative_token_samples]

    # Combined stats
    all_lengths = positive_lengths + negative_lengths

    return {
        'token_stats': {
            'combined': {
                'average_tokens': sum(all_lengths) / len(all_lengths) if all_lengths else 0,
                'min_tokens': min(all_lengths) if all_lengths else 0,
                'max_tokens': max(all_lengths) if all_lengths else 0,
                'total_tokens': sum(all_lengths),
            },
            'positive': {
                'average_tokens': sum(positive_lengths) / len(positive_lengths) if positive_lengths else 0,
                'min_tokens': min(positive_lengths) if positive_lengths else 0,
                'max_tokens': max(positive_lengths) if positive_lengths else 0,
                'total_tokens': sum(positive_lengths),
            },
            'negative': {
                'average_tokens': sum(negative_lengths) / len(negative_lengths) if negative_lengths else 0,
                'min_tokens': min(negative_lengths) if negative_lengths else 0,
                'max_tokens': max(negative_lengths) if negative_lengths else 0,
                'total_tokens': sum(negative_lengths),
            },
        }
    }


def decode_batch(args):
    """Helper function to decode a batch of tokens using a tokenizer."""
    samples, tokenizer = args
    return tokenizer.batch_decode(samples, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def parallel_decode(tokenized_samples, tokenizer, num_workers=None, batch_size=1000):
    """
    Parallelized batch decoding using multiprocessing.

    Args:
        tokenized_samples: List of tokenized samples to decode.
        tokenizer: Tokenizer object used for decoding.
        num_workers: Number of worker processes to use.
        batch_size: Number of samples per batch.

    Returns:
        List of decoded text samples.
    """
    if num_workers is None:
        num_workers = cpu_count()
    # Split input into batches
    batches = [tokenized_samples[i : i + batch_size] for i in range(0, len(tokenized_samples), batch_size)]
    # Prepare arguments for parallel processing
    args = [(batch, tokenizer) for batch in batches]
    # Use multiprocessing pool to decode in parallel
    with Pool(processes=num_workers) as pool:
        results = pool.map(decode_batch, args)
    # Flatten the result list
    return [item for sublist in results for item in sublist]


def main():
    """Main function to build coherent dataset."""
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    logger = setup_logger()

    logger.info(f"Loading tokenizer {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    logger.info('Loading dataset OpenR1-Math-220k')
    ds = load_dataset('open-r1/OpenR1-Math-220k', 'default')

    # Apply max_size limit and shuffle
    ds = ds['train'].shuffle().select(range(min(args.max_size, len(ds['train']))))
    # Convert dataset to list if it's not already
    dataset = list(ds)

    if args.load_custom_source:
        logger.info('Loading custom dataset...')
        try:
            custom_dataset = load_from_jsonl_file(args.load_custom_source)
            if len(custom_dataset) > args.max_size:
                random.shuffle(custom_dataset)
                custom_dataset = custom_dataset[: args.max_size]
            logger.info(f"Loaded custom {len(custom_dataset)} items")
            dataset = custom_dataset + dataset
        except Exception:
            pass

    # Extract and tokenize positive samples
    logger.info('Extracting and tokenizing positive samples from dataset')
    start_time = time.time()
    positive_token_samples = extract_positive_samples(dataset, tokenizer, args.max_tokens)
    logger.info(f"Extracted {len(positive_token_samples)} positive token samples in {time.time() - start_time:.2f} seconds")

    # Precompute noise tokens
    logger.info('Precomputing noise tokens')
    noise_tokens = precompute_noise_tokens(tokenizer)

    # Create incoherent token samples
    logger.info('Creating negative samples...')
    start_time = time.time()
    negative_token_samples = extract_negative_samples(
        positive_token_samples,
        vocab_size=tokenizer.vocab_size,
        noise_tokens=noise_tokens,
        max_tokens=args.max_tokens,
    )
    logger.info(f"Created {len(negative_token_samples)} negative samples in {time.time() - start_time:.2f} seconds")

    # Decode all samples back to text
    logger.info('Decoding positive token samples back to text...')
    positive_samples = parallel_decode(positive_token_samples, tokenizer)

    logger.info('Decoding negative token samples back to text...')
    negative_samples = parallel_decode(negative_token_samples, tokenizer)

    logger.info('Creating repetition negative samples...')
    selected_positive_samples = random.choices(positive_samples, k=int(0.1 * len(positive_samples)))
    repetition_negative_samples = create_repetition_negative_samples(selected_positive_samples, 5, 15)
    logger.info(f"Created {len(repetition_negative_samples)} repetition negative samples")
    negative_samples.extend(repetition_negative_samples)

    # Create dataset dictionaries
    coherent_samples = [{'text': sample, 'label': 1} for sample in positive_samples]
    incoherent_samples = [{'text': sample, 'label': 0} for sample in negative_samples]

    # Shuffle and merge samples
    merged_samples = coherent_samples + incoherent_samples
    random.shuffle(merged_samples)

    # Calculate basic statistics
    stats = {
        'total_samples': len(merged_samples),
        'coherent_samples': len(coherent_samples),
        'incoherent_samples': len(incoherent_samples),
        'train_samples': int(args.split_ratio * len(merged_samples)),
        'test_samples': len(merged_samples) - int(args.split_ratio * len(merged_samples)),
        'max_tokens_per_sample': args.max_tokens,
    }

    # Split and save datasets
    logger.info(f"Splitting coherent dataset into train/test with ratio {args.split_ratio}")
    split = int(args.split_ratio * len(merged_samples))
    train_samples = merged_samples[:split]
    test_samples = merged_samples[split:]

    # Save files
    logger.info(f"Saving coherent dataset to {args.save_dir}")
    save_to_parquet_file(train_samples, f"{args.save_dir}/train.parquet", compression='snappy')
    save_to_parquet_file(test_samples, f"{args.save_dir}/test.parquet", compression='snappy')
    save_to_json_file(stats, f"{args.save_dir}/metadata.json")
    logger.info('Dataset creation completed successfully')


if __name__ == '__main__':
    main()
