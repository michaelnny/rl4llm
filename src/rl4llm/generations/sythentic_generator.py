import datetime
import json
import logging
import math
import os
import random
from typing import Any, Dict, List, Tuple

import pandas as pd
import torch
from tqdm import tqdm
from transformers import LogitsProcessor, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class DegradingLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        max_length: int,
        prompt_length: int = 0,
        degradation_factor: float = 0.5,
        top_k: int = 50,
        eos_token_id: int = None,
        min_degradation_strength: float = 0.7,
        min_coherent_tokens: int = 100,
    ):
        """
        Logits processor with fast, exponential degradation strength.

        Args:
            max_length (int): Total length of the generation.
            prompt_length (int): Length of the input prompt.
            degradation_factor (float): Factor to scale logits dampening (0.0–1.0).
            top_k (int): Number of top tokens to sample from randomly.
            eos_token_id (int): ID of the EOS token to mask until coherent_length.
            min_degradation_strength (float): Minimum strength once degradation starts.
        """
        self.max_length = max_length
        self.prompt_length = prompt_length
        self.degradation_factor = degradation_factor
        self.top_k = top_k
        self.eos_token_id = eos_token_id
        self.min_degradation_strength = min_degradation_strength
        self.min_coherent_tokens = min_coherent_tokens

        # Randomize coherent length uniformly
        coherent_fraction = random.uniform(0.05, 0.3)  # if max_length > 1000 else random.uniform(0.01, 0.1)
        self.coherent_length = max(min_coherent_tokens, int(max_length * coherent_fraction))

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits with EOS masking and mixed top-K/inverse sampling.

        Args:
            input_ids (torch.LongTensor): Generated token IDs so far [batch_size, sequence_length].
            scores (torch.FloatTensor): Logits for next token [batch_size, vocab_size].

        Returns:
            torch.FloatTensor: Modified logits.
        """
        current_length = input_ids.shape[-1] - self.prompt_length

        # Calculate effective maximum length
        effective_max_length = self.max_length - self.prompt_length

        # Determine if we're in the final portion of generation
        near_end = current_length >= (effective_max_length * 0.75)

        # Mask EOS token with modified logic
        if self.eos_token_id is not None:
            # Only mask EOS during early generation or if no degradation applied yet
            if current_length < self.coherent_length:
                scores[:, self.eos_token_id] = -1e8

        # Calculate degradation strength with improved logic
        if current_length <= self.coherent_length and not near_end:
            # No degradation during initial coherent part, unless near the end
            return scores
        else:
            # Ensure at least some degradation on short outputs
            if effective_max_length <= self.coherent_length or near_end:
                # We're either in a very short output scenario or near the end - force degradation
                degradation_strength = 0.8 if not near_end else 1.0
            else:
                # Standard exponential degradation
                progress = (current_length - self.coherent_length) / (effective_max_length - self.coherent_length)
                degradation_strength = self.min_degradation_strength + (1.0 - self.min_degradation_strength) * (
                    1.0 - math.exp(-5.0 * progress)
                )
                degradation_strength = min(1.0, degradation_strength)

            if random.random() < degradation_strength:
                # Blend between strategies based on length
                if random.random() < 0.5 if current_length < self.coherent_length * 2 else 0.7:
                    # Top-K sampling
                    k_value = max(5, self.top_k)  # Ensure k is not too small
                    values, indices = torch.topk(scores, k_value, dim=-1)
                    mask = torch.ones_like(scores, dtype=torch.bool)
                    mask.scatter_(-1, indices, False)
                    scores[mask] = -float('inf')
                else:
                    # Inverse-weighted sampling with boosted blend factor
                    inverted_scores = -scores
                    blend_factor = min(1.0, self.degradation_factor * degradation_strength)
                    scores = (1 - blend_factor) * scores + blend_factor * inverted_scores

        return scores


class SyntheticDataGenerator:
    """
    A generator that produces both coherent (positive) and incoherent (negative) samples for training or evaluation.
    Supports calling generate_dataset repeatedly (each for train or eval data)
    and then combining any intermediate checkpoint files plus remaining samples when close() is called.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = './data',
        max_attempts: int = 3,
        log_level: int = logging.INFO,
        checkpoint_interval: int = 1000,  # save a checkpoint every N samples
    ):
        """
        Args:
          model: The pre-trained model used for generation.
          tokenizer: Model tokenizer.
          device (str): 'cuda' or 'cpu'.
          output_dir (str): Base directory where generated files will be saved.
          max_attempts (int): Number of attempts (in case generation fails) per prompt.
          log_level (int): Logger level.
          checkpoint_interval (int): Save intermediate checkpoint if samples collected > this number.
        """
        self.device = device
        self.max_attempts = max_attempts
        self.checkpoint_interval = checkpoint_interval

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        self.output_dir = run_dir

        logger.setLevel(log_level)
        self.model = model.to(self.device).eval()
        self.tokenizer = tokenizer
        self.model_name = model.config.name_or_path
        self.vocab_size = tokenizer.vocab_size

        # If no explicit pad token, set pad to eos.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.run_metadata = {
            'model_name': self.model_name,
            'timestamp': datetime.datetime.now().isoformat(),
        }

        # Data accumulators – these lists persist over multiple generate_dataset() calls.
        self.train_samples: List[Dict] = []
        self.val_samples: List[Dict] = []
        self.train_count = 0
        self.val_count = 0

        # Track checkpoint files and counters so they are not overwritten
        self.train_checkpoint_files: List[str] = []
        self.val_checkpoint_files: List[str] = []
        self.train_checkpoint_counter = 0
        self.val_checkpoint_counter = 0

    def _build_prompt(self, prompt: str, system_prompt: str = None) -> str:
        """Create a prompt string (using chat formatting if a system prompt exists)."""
        if system_prompt:
            messages = [
                {'role': 'system', 'content': system_prompt.strip()},
                {'role': 'user', 'content': prompt.strip()},
            ]
        else:
            messages = [{'role': 'user', 'content': prompt.strip()}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _try_generate(self, generate_func, *args, **kwargs) -> List[Tuple[str, Dict]]:
        """Helper for retrying generation up to max_attempts."""
        for attempt in range(self.max_attempts):
            try:
                samples = generate_func(*args, **kwargs)
                if samples and any(sample[0].strip() for sample in samples):
                    return samples
            except Exception as e:
                if attempt == self.max_attempts - 1:
                    logger.warning(f"Failed after {self.max_attempts} attempts: {e}")
                    return []
                # Otherwise, simply try again.
                continue
        return []

    def generate_positive_samples(
        self,
        prompt: str,
        system_prompt: str = None,
        n_samples: int = 1,
        min_new_tokens: int = 30,
        max_new_tokens: int = 1024,
    ) -> List[Tuple[str, Dict]]:
        """
        Generate coherent samples from a prompt.
        Returns: list of tuples (completion text, stats dict)
        """
        gen_params = {
            'min_new_tokens': min_new_tokens,
            'max_new_tokens': max_new_tokens,
            'temperature': 0.7,
            'top_p': 0.9,
            'repetition_penalty': 1.1,
            'do_sample': True,
            'pad_token_id': self.tokenizer.pad_token_id,
            'num_return_sequences': n_samples,
        }
        input_prompt = self._build_prompt(prompt, system_prompt)
        inputs = self.tokenizer(input_prompt, return_tensors='pt').to(self.device)
        prompt_tokens = len(inputs.input_ids[0])
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)
        samples = []
        for output in outputs:
            # Skip prompt tokens and decode the remainder.
            completion_tokens = output[prompt_tokens:]
            completion = self.tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
            stats = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': len(completion_tokens),
                'total_tokens': prompt_tokens + len(completion_tokens),
            }
            samples.append((completion, stats))
        return samples

    def generate_negative_samples(
        self,
        prompt: str,
        system_prompt: str = None,
        n_samples: int = 1,
        min_new_tokens: int = 30,
        max_new_tokens: int = 1024,
    ) -> List[Tuple[str, Dict]]:
        """
        Generate incoherent samples from a prompt.
        Returns: list of tuples (completion text, stats dict)
        """
        input_prompt = self._build_prompt(prompt, system_prompt)
        inputs = self.tokenizer(input_prompt, return_tensors='pt').to(self.device)
        prompt_tokens = len(inputs.input_ids[0])
        # For negative samples, choose a random max tokens and add a logits processor.
        max_tokens = random.choice(range(int(max_new_tokens * 0.6), max_new_tokens))
        gen_params = {
            'min_new_tokens': min_new_tokens,
            'max_new_tokens': max_tokens,
            'temperature': 1.0,
            'top_p': 1.0,
            'do_sample': True,
            'repetition_penalty': 1.0,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'num_return_sequences': n_samples,
            'logits_processor': [
                DegradingLogitsProcessor(
                    max_length=max_new_tokens,
                    prompt_length=prompt_tokens,
                    degradation_factor=random.uniform(0.4, 0.8),
                    top_k=random.randint(int(self.vocab_size // 2), self.vocab_size),
                    eos_token_id=self.tokenizer.eos_token_id,
                    min_degradation_strength=0.7,
                    min_coherent_tokens=min_new_tokens,
                )
            ],
        }
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)
        negative_samples = []
        for output in outputs:
            completion_tokens = output[prompt_tokens:]
            completion = self.tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
            stats = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': len(completion_tokens),
                'total_tokens': prompt_tokens + len(completion_tokens),
            }
            negative_samples.append((completion, stats))
        return negative_samples

    def _save_checkpoint(self, is_train: bool):
        """
        Save the current samples in memory to a new checkpoint file and clear the list.
        The checkpoint file name uses an internal counter so that multiple checkpoints are not overwritten.
        """
        dataset_type = 'train' if is_train else 'val'
        samples = self.train_samples if is_train else self.val_samples
        if not samples:
            return

        if is_train:
            counter = self.train_checkpoint_counter
        else:
            counter = self.val_checkpoint_counter

        checkpoint_file = os.path.join(self.output_dir, f"{dataset_type}_checkpoint_{counter}.jsonl.gz")
        df = pd.DataFrame(samples)
        df.to_json(checkpoint_file, orient='records', lines=True, compression='gzip')

        # Record the checkpoint and clear the memory.
        if is_train:
            self.train_checkpoint_files.append(checkpoint_file)
            self.train_checkpoint_counter += 1
            self.train_samples.clear()
        else:
            self.val_checkpoint_files.append(checkpoint_file)
            self.val_checkpoint_counter += 1
            self.val_samples.clear()
        logger.info(f"Saved checkpoint with {len(df)} {dataset_type} samples to {checkpoint_file}")

    def generate_dataset(
        self,
        input_data: List[Dict[str, Any]],
        system_prompt: str = None,
        n_samples: int = 5,
        min_new_tokens: int = 30,
        max_new_tokens: int = 1024,
        positive_only: bool = False,
        is_train: bool = True,
    ) -> None:
        """
        Process a list of input items (each with at least a "question" field) and generate samples.
        Append generated samples (with stats and an “is_coherent” flag) to internal lists.
        Checkpoints are automatically saved if the number of samples exceeds checkpoint_interval.
        """
        dataset_type = 'train' if is_train else 'val'
        for item in tqdm(input_data, desc=f"Generating {dataset_type} samples"):
            prompt = item['question']

            # Generate positive (coherent) samples.
            positive_samples = self._try_generate(
                self.generate_positive_samples,
                prompt,
                system_prompt,
                n_samples,
                min_new_tokens,
                max_new_tokens,
            )
            for text, stats in positive_samples:
                if not text.strip():
                    continue
                output_item = item.copy()
                output_item.update(
                    {
                        'completion': text,
                        'is_coherent': True,
                        'model_name': self.model_name,
                        'prompt_tokens': stats['prompt_tokens'],
                        'completion_tokens': stats['completion_tokens'],
                        'total_tokens': stats['total_tokens'],
                    }
                )
                if is_train:
                    self.train_samples.append(output_item)
                    self.train_count += 1
                else:
                    self.val_samples.append(output_item)
                    self.val_count += 1

            # Optionally generate negative (incoherent) samples.
            if not positive_only:
                # Slightly more negatives may be generated.
                n_neg = int(n_samples * 1.1)
                negative_samples = self._try_generate(
                    self.generate_negative_samples,
                    prompt,
                    system_prompt,
                    n_neg,
                    min_new_tokens,
                    max_new_tokens,
                )
                for text, stats in negative_samples:
                    if not text.strip():
                        continue
                    output_item = item.copy()
                    output_item.update(
                        {
                            'completion': text,
                            'is_coherent': False,
                            'model_name': self.model_name,
                            'prompt_tokens': stats['prompt_tokens'],
                            'completion_tokens': stats['completion_tokens'],
                            'total_tokens': stats['total_tokens'],
                        }
                    )
                    if is_train:
                        self.train_samples.append(output_item)
                        self.train_count += 1
                    else:
                        self.val_samples.append(output_item)
                        self.val_count += 1

            # Check if its time to write a checkpoint to reduce memory usage.
            current_count = len(self.train_samples) if is_train else len(self.val_samples)
            if current_count >= self.checkpoint_interval:
                self._save_checkpoint(is_train)

        logger.info(f"Total {dataset_type} samples generated so far: {self.train_count if is_train else self.val_count}")

    def close(self):
        """
        Called after all generate_dataset() calls.
        Combines any checkpoint files with samples remaining in memory,
        post-processes and shuffles the data for train and eval,
        writes out final datasets and metadata.
        """
        for dataset_type, mem_samples, checkpoint_files in [
            ('train', self.train_samples, self.train_checkpoint_files),
            ('val', self.val_samples, self.val_checkpoint_files),
        ]:
            # Combine checkpoint files and samples in memory.
            all_samples = []
            for fname in checkpoint_files:
                try:
                    df_chk = pd.read_json(fname, lines=True, compression='gzip')
                    all_samples.extend(df_chk.to_dict('records'))
                except Exception as e:
                    logger.error(f"Error reading {fname}: {e}")

            # Also include any samples remaining in memory.
            all_samples.extend(mem_samples)

            if not all_samples:
                logger.info(f"No {dataset_type} samples to save.")
                continue

            df_all = pd.DataFrame(all_samples)

            # Optionally filter out especially short incoherent samples.
            if 'is_coherent' in df_all.columns:
                incoherent_mask = ~df_all['is_coherent']
                if incoherent_mask.any():
                    # Filter out bottom 25% by completion tokens.
                    cutoff = df_all.loc[incoherent_mask, 'completion_tokens'].quantile(0.25)
                    df_all = df_all[~(incoherent_mask & (df_all['completion_tokens'] <= cutoff))]

            # Shuffle the dataset.
            df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)

            final_file = os.path.join(self.output_dir, f"{dataset_type}.jsonl.gz")
            df_all.to_json(final_file, orient='records', lines=True, compression='gzip')
            logger.info(f"Saved final {dataset_type} dataset with {len(df_all)} samples to {final_file}")

            # Record dataset stats into metadata.
            if 'is_coherent' in df_all.columns:
                n_coherent = int(df_all['is_coherent'].sum())
                n_total = len(df_all)
            else:
                n_coherent = n_total = len(df_all)
            self.run_metadata[dataset_type] = {
                'coherent': n_coherent,
                'incoherent': n_total - n_coherent,
                'total_samples': n_total,
                'avg_prompt_tokens': float(df_all['prompt_tokens'].mean()),
                'avg_completion_tokens': float(df_all['completion_tokens'].mean()),
                'min_completion_tokens': int(df_all['completion_tokens'].min()),
                'max_completion_tokens': int(df_all['completion_tokens'].max()),
                'p25_completion_tokens': float(df_all['completion_tokens'].quantile(0.25)),
                'p50_completion_tokens': float(df_all['completion_tokens'].quantile(0.50)),
                'p75_completion_tokens': float(df_all['completion_tokens'].quantile(0.75)),
                'p90_completion_tokens': float(df_all['completion_tokens'].quantile(0.90)),
                'avg_total_tokens': float(df_all['total_tokens'].mean()),
                'max_total_tokens': int(df_all['total_tokens'].max()),
            }

        # Save metadata (using json.dump correctly).
        meta_file = os.path.join(self.output_dir, 'metadata.json')
        with open(meta_file, 'w') as f:
            json.dump(self.run_metadata, f, indent=2)
        logger.info(f"Saved metadata to {meta_file}")
        logger.info(f"Data generation complete. Files saved to {self.output_dir}")


# Example usage
if __name__ == '__main__':

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = 'Qwen/Qwen2.5-0.5B-Instruct'

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    torch_dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch_dtype).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    generator = SyntheticDataGenerator(
        model=model, tokenizer=tokenizer, device=device, output_dir='./test_data/coherence_samples'
    )

    # Example 1: Generate from a list of prompts
    prompts = [
        'Explain how photosynthesis works.',
        'Write a short story about a detective solving a mystery.',
        'Describe the process of making sourdough bread.',
    ]
    dataset = generator.generate_dataset(prompts, n_samples=5, max_length=512)

    print(len(dataset))

    # Example 2: Generate from a CSV file
    # generator.generate_dataset_from_file(
    #     "prompts.csv",
    #     prompt_column="question",
    #     n_samples=3,
    #     file_format="csv"
    # )
