import datetime
import json
import logging
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from pydantic import BaseModel, Field
from tqdm import tqdm
from transformers import LogitsProcessor, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)


class GeneratedSample(BaseModel):
    """
    Data structure for storing generated samples with minimal metadata.
    """

    prompt: str = Field(..., description='The input prompt used to generate the sample')
    completion: str = Field(..., description='The generated text output from the model')
    is_coherent: bool = Field(..., description='Whether the sample is coherent (True) or degraded (False)')
    model_name: str = Field(..., description='Name of the model used for generation')
    prompt_tokens: int = Field(..., description='Number of tokens in the prompt')
    completion_tokens: int = Field(..., description='Number of tokens in the generated completion')
    total_tokens: int = Field(..., description='Total number of tokens (prompt + completion)')

    class Config:
        arbitrary_types_allowed = True


class GenerationMetadata(BaseModel):
    """
    Metadata for a generation run.
    """

    model_name: str = Field(..., description='Name of the model used for generation')
    timestamp: str = Field(..., description='Timestamp of the generation run')
    train_ratio: float = Field(..., description='Ratio of data used for training vs validation')
    train_statistics: Dict[str, Any] = Field(..., description='Statistics about the training dataset')
    val_statistics: Dict[str, Any] = Field(..., description='Statistics about the validation dataset')


class DegradingLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        max_length: int,
        prompt_length: int = 0,  # Length of input prompt
        degradation_factor: float = 0.5,  # Controls dampening strength
        top_k: int = 50,  # Number of top tokens for random sampling
        eos_token_id: int = None,  # EOS token ID to mask
        min_degradation_strength: float = 0.6,  # Starting strength after coherent_length
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

        # Randomize coherent length uniformly
        coherent_fraction = random.uniform(0.05, 0.25) if max_length > 1500 else random.uniform(0.1, 0.3)
        self.coherent_length = int(max_length * coherent_fraction)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """
        Process logits with EOS masking and mixed top-K/inverse sampling.

        Args:
            input_ids (torch.LongTensor): Generated token IDs so far [batch_size, sequence_length].
            scores (torch.FloatTensor): Logits for next token [batch_size, vocab_size].

        Returns:
            torch.FloatTensor: Modified logits.
        """
        current_length = input_ids.shape[-1]

        # Mask EOS token before coherent_length to ensure sequence continues
        if self.eos_token_id is not None and current_length < self.coherent_length:
            scores[:, self.eos_token_id] = -1e8

        # Calculate degradation strength
        if current_length <= self.coherent_length:
            return scores  # No change during coherent part
        else:
            # Scale based on remaining length after prompt and coherent part
            effective_max_length = self.max_length - self.prompt_length
            if effective_max_length <= self.coherent_length:
                degradation_strength = 1.0  # Max degradation if no room left
            else:
                # Exponential degradation: starts at min_degradation_strength, quickly reaches 1.0
                progress = (current_length - self.coherent_length) / (effective_max_length - self.coherent_length)
                degradation_strength = self.min_degradation_strength + (1.0 - self.min_degradation_strength) * (
                    1.0 - math.exp(-5.0 * progress)
                )
                degradation_strength = min(1.0, degradation_strength)  # Cap at 1.0

        # Apply degradation with probability tied to degradation_strength
        if random.random() < degradation_strength:
            # Randomly choose between top-K sampling and inverse-weighted sampling
            if random.random() < 0.5:  # 50% chance for top-K
                # Top-K sampling: Zero out all but top K logits
                values, indices = torch.topk(scores, self.top_k, dim=-1)
                mask = torch.ones_like(scores, dtype=torch.bool)
                mask.scatter_(-1, indices, False)
                scores[mask] = -float('inf')
            else:  # 50% chance for inverse-weighted sampling
                # Blend original and inverted logits
                inverted_scores = -scores
                blend_factor = self.degradation_factor * degradation_strength
                scores = (1 - blend_factor) * scores + blend_factor * inverted_scores

        return scores


class SyntheticDataGenerator:
    """
    Generates coherent and incoherent data for training classifier models.
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        system_prompt: str = None,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        output_dir: str = './data',
        max_attempts: int = 3,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the synthetic data generator.

        Args:
            model: Pre-trained model for generation.
            tokenizer: Tokenizer for the model.
            system_prompt: System prompt to prepend to all generated samples.
            device (str): Device to run the model on ('cuda' or 'cpu').
            output_dir (str): Directory to save generated data.
            max_attempts (int): Maximum number of generation attempts before skipping.
            log_level (int): Logging level (default: INFO).
        """

        self.device = device
        self.system_prompt = system_prompt
        self.max_attempts = max_attempts

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        self.output_dir = run_dir

        # Set logger level
        logger.setLevel(log_level)

        # # Create output directory if it doesn't exist
        # os.makedirs(output_dir, exist_ok=True)

        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.model_name = model.config.name_or_path
        self.vocab_size = tokenizer.vocab_size

        # If the model doesn't have a padding token, set it to the eos token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Track metadata for the generation runs
        self.run_metadata = {
            'model_name': self.model_name,
            'timestamp': datetime.datetime.now().isoformat(),
        }

    def generate_good_samples(self, prompt: str, n_samples: int = 1, max_new_tokens: int = 1024) -> List[Tuple[str, Dict]]:
        """
        Generate multiple coherent (good) samples from a single prompt in one call.

        Args:
            prompt (str): The input prompt.
            n_samples (int): Number of samples to generate.
            max_new_tokens (int): Maximum length of the generated text in tokens.

        Returns:
            List[Tuple[str, Dict]]: List of tuples, each containing the generated text (completion) and a dictionary of generation parameters.
        """
        gen_params = {
            'max_new_tokens': max_new_tokens,
            'temperature': 0.7,
            'top_p': 0.9,
            'do_sample': True,
            'pad_token_id': self.tokenizer.pad_token_id,
            'num_return_sequences': n_samples,
        }

        input_prompt = self._build_prompt(prompt)
        inputs = self.tokenizer(input_prompt, return_tensors='pt').to(self.device)
        prompt_tokens = len(inputs.input_ids[0])

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)

        samples = []
        for output in outputs:
            # Skip the prompt tokens and decode the remaining tokens for the completion
            completion_tokens = output[prompt_tokens:]
            completion = self.tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
            completion_tokens = len(completion_tokens)
            stats = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
            }
            samples.append((completion, stats))

        return samples

    def generate_bad_samples(self, prompt: str, n_samples: int = 1, max_new_tokens: int = 1024) -> List[Tuple[str, Dict]]:
        """
        Generate multiple incoherent (bad) samples from a single prompt in a batched manner.

        This method first generates a batch of coherent parts using num_return_sequences, then for each coherent text
        generates a degraded continuation using individually randomized parameters.

        Args:
            prompt (str): The input prompt.
            n_samples (int): Number of samples to generate.
            max_new_tokens (int): Maximum length for the entire generated text (coherent + degraded).

        Returns:
            List[Tuple[str, Dict]]: List of tuples, each containing the full generated text, a dictionary of generation parameters.
        """

        # Prepare inputs
        input_prompt = self._build_prompt(prompt)
        inputs = self.tokenizer(input_prompt, return_tensors='pt').to(self.device)
        prompt_tokens = len(inputs.input_ids[0])

        # Define generation parameters
        max_tokens = random.choice(range(int(max_new_tokens * 0.6), max_new_tokens))
        gen_params = {
            'max_new_tokens': max_tokens,
            'temperature': 1.0,  # Base temp; logits processor drives chaos
            'top_p': 1.0,  # Full distribution sampling
            'do_sample': True,
            'eos_token_id': self.tokenizer.eos_token_id,
            'pad_token_id': self.tokenizer.pad_token_id,
            'num_return_sequences': n_samples,
            'logits_processor': [
                DegradingLogitsProcessor(
                    max_length=max_new_tokens,
                    prompt_length=prompt_tokens,
                    degradation_factor=random.uniform(0.5, 0.9),
                    top_k=random.randint(int(self.vocab_size // 2), self.vocab_size),  # Randomize top-K range
                    eos_token_id=self.tokenizer.eos_token_id,  # Pass EOS token ID
                    min_degradation_strength=0.7,  # Start strong after coherent part
                )
            ],
        }

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)

        # Process outputs
        bad_samples = []
        for output in outputs:
            # Skip the prompt tokens and decode the remaining tokens for the completion
            completion_tokens = output[prompt_tokens:]
            completion = self.tokenizer.decode(completion_tokens, skip_special_tokens=True).strip()
            completion_tokens = len(completion_tokens)
            stats = {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': prompt_tokens + completion_tokens,
            }

            bad_samples.append((completion, stats))

        return bad_samples

    def _build_prompt(self, prompt: str) -> str:
        """Builds model input prompt from user input prompt."""
        if self.system_prompt:
            messages = [{'role': 'system', 'content': self.system_prompt}, {'role': 'user', 'content': prompt}]
        else:
            messages = [{'role': 'user', 'content': prompt}]

        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def generate_dataset(
        self, prompts: List[str], n_samples_per_prompt: int = 5, max_new_tokens: int = 1024, is_train: bool = True
    ) -> pd.DataFrame:
        """
        Generate a dataset of good and bad samples from a list of prompts.

        Args:
            prompts (List[str]): List of prompts.
            n_samples_per_prompt (int): Number of samples per prompt.
            max_length (int): Maximum length of generated text in tokens.

        Returns:
            pd.DataFrame: DataFrame containing all generated samples.
        """
        all_samples = []

        for prompt in tqdm(prompts, desc='Generating samples from prompts'):
            try:
                # Generate good samples in one batch
                for attempt in range(self.max_attempts):
                    try:
                        good_samples = self.generate_good_samples(prompt, n_samples_per_prompt, max_new_tokens)
                        if any(good_text.strip() for good_text, _ in good_samples):
                            break
                    except Exception as e:
                        if attempt == self.max_attempts - 1:
                            logger.warning(f"Failed to generate good samples after {self.max_attempts} attempts: {e}")
                        continue

                for good_text, good_stats in good_samples:
                    good_sample = GeneratedSample(
                        prompt=prompt,
                        completion=good_text,
                        is_coherent=True,
                        model_name=self.model_name,
                        prompt_tokens=good_stats.get('prompt_tokens', 0),
                        completion_tokens=good_stats.get('completion_tokens', 0),
                        total_tokens=good_stats.get('total_tokens', 0),
                    )
                    all_samples.append(good_sample.model_dump())

                # Generate bad samples in one batch
                for attempt in range(self.max_attempts):
                    try:
                        bad_samples = self.generate_bad_samples(prompt, n_samples_per_prompt, max_new_tokens)
                        if any(bad_text.strip() for bad_text, _ in bad_samples):
                            break
                    except Exception as e:
                        if attempt == self.max_attempts - 1:
                            logger.warning(f"Failed to generate bad samples after {self.max_attempts} attempts: {e}")
                        continue

                for bad_text, bad_stats in bad_samples:
                    bad_sample = GeneratedSample(
                        prompt=prompt,
                        completion=bad_text,
                        is_coherent=False,
                        model_name=self.model_name,
                        prompt_tokens=bad_stats.get('prompt_tokens', 0),
                        completion_tokens=bad_stats.get('completion_tokens', 0),
                        total_tokens=bad_stats.get('total_tokens', 0),
                    )
                    all_samples.append(bad_sample.model_dump())

            except Exception as e:
                logger.error(f"Error generating samples for prompt '{prompt[:50]}...': {e}")
                continue
        if len(all_samples) == 0:
            logger.error('No samples generated')
            return
        df = pd.DataFrame(all_samples)
        self.save_dataset(df, is_train)

    def save_dataset(self, df: pd.DataFrame, is_train: bool = True) -> None:
        """
        Save the generated dataset along with metadata and statistics.

        Args:
            df (pd.DataFrame): DataFrame containing the generated samples.
            train_ratio (float): Ratio of data to use for training vs validation.
        """
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # Shuffle the DataFrame
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        # Save datasets
        prefix = 'train' if is_train else 'val'
        save_file = os.path.join(self.output_dir, f"{prefix}.jsonl.gz")

        df.to_json(save_file, orient='records', lines=True, compression='gzip')

        # Calculate statistics
        stats = {
            'model_name': self.model_name,
            'timestamp': timestamp,
            'coherent': int(df['is_coherent'].sum()),
            'incoherent': int(len(df) - df['is_coherent'].sum()),
            'total_samples': len(df),
            'avg_prompt_tokens': float(df['prompt_tokens'].mean()),
            'avg_completion_tokens': float(df['completion_tokens'].mean()),
            'avg_total_tokens': float(df['total_tokens'].mean()),
            'max_tokens': int(df['total_tokens'].max()),
        }

        with open(os.path.join(self.output_dir, f"{prefix}_metadata.json"), 'w') as f:
            f.write(json.dumps(stats, indent=2))

        logger.info(f"Saved dataset to {self.output_dir}:")


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
    dataset = generator.generate_dataset(prompts, n_samples_per_prompt=5, max_length=512)

    print(len(dataset))

    # Example 2: Generate from a CSV file
    # generator.generate_dataset_from_file(
    #     "prompts.csv",
    #     prompt_column="question",
    #     n_samples_per_prompt=3,
    #     file_format="csv"
    # )
