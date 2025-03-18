import datetime
import hashlib
import json
import logging
import math
import multiprocessing as mp
import os
import random
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from pydantic import BaseModel, Field
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("synthetic_data_generator.log")],
)
logger = logging.getLogger("SyntheticDataGenerator")


class GeneratedSample(BaseModel):
    """
    Data structure for storing generated samples with minimal metadata.
    """

    prompt: str = Field(..., description="The input prompt used to generate the sample")
    text: str = Field(..., description="The generated text output from the model")
    is_coherent: bool = Field(..., description="Whether the sample is coherent (True) or degraded (False)")
    model_name: str = Field(..., description="Name of the model used for generation")
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the generated completion")
    total_tokens: int = Field(..., description="Total number of tokens (prompt + completion)")
    degradation_methods: List[str] = Field(
        default_factory=list, description="List of degradation methods applied (for incoherent samples)"
    )
    sample_id: str = Field("", description="Unique identifier for the sample")

    class Config:
        arbitrary_types_allowed = True

    def __init__(self, **data):
        """
        Initialize a GeneratedSample instance.
        If no sample_id is provided, it creates one based on a hash of prompt and text.
        """
        super().__init__(**data)
        if not self.sample_id:
            content_hash = hashlib.md5(f"{self.prompt}{self.text}".encode()).hexdigest()
            self.sample_id = f"{'coherent' if self.is_coherent else 'incoherent'}_{content_hash[:10]}"


class GenerationMetadata(BaseModel):
    """
    Metadata for a generation run.
    """

    model_name: str = Field(..., description="Name of the model used for generation")
    timestamp: str = Field(..., description="Timestamp of the generation run")
    device: str = Field(..., description="Device used for generation (cuda/cpu)")
    seed: int = Field(..., description="Random seed used for reproducibility")
    train_ratio: float = Field(..., description="Ratio of data used for training vs validation")
    train_statistics: Dict[str, Any] = Field(..., description="Statistics about the training dataset")
    val_statistics: Dict[str, Any] = Field(..., description="Statistics about the validation dataset")
    degradation_method_counts: Dict[str, int] = Field(..., description="Count of each degradation method used")


class DegradingLogitsProcessor(LogitsProcessor):

    def __init__(
        self,
        max_length: int,
        prompt_length: int = 0,          # Length of input prompt
        degradation_factor: float = 0.5, # Controls dampening strength
        top_k: int = 50,                 # Number of top tokens for random sampling
        eos_token_id: int = None,        # EOS token ID to mask
        min_degradation_strength: float = 0.7,  # Starting strength after coherent_length
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
        coherent_fraction = random.uniform(0.05, 0.3)
        if prompt_length > 0:
            min_coherent = min(prompt_length + int(max_length * 0.1), int(max_length * 0.3))
            self.coherent_length = max(min_coherent, int(max_length * coherent_fraction))
        else:
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
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        output_dir: str = "./data",
        max_attempts: int = 3,
        log_level: int = logging.INFO,
        num_workers: Optional[int] = None,  # For the reusable pool
    ):
        """
        Initialize the synthetic data generator.

        Args:
            model_name (str): Name of the model from Hugging Face to use for generation.
            device (str): Device to run the model on ('cuda' or 'cpu').
            seed (int): Random seed for reproducibility.
            output_dir (str): Directory to save generated data.
            max_attempts (int): Maximum number of generation attempts before skipping.
            log_level (int): Logging level (default: INFO).
            num_workers (int): Number of worker threads for post-generation processing.
        """
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        self.max_attempts = max_attempts
        self.num_workers = num_workers if num_workers and num_workers > 0 else mp.cpu_count()

        # Set logger level
        logger.setLevel(log_level)

        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Set seeds for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        logger.info(f"Loading model {model_name}...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            ).to(device)
            logger.info("Model loaded successfully!")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

        # If the model doesn't have a padding token, set it to the eos token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Track metadata for the generation runs
        self.run_metadata = {
            "model_name": model_name,
            "timestamp": datetime.datetime.now().isoformat(),
            # "device": device,
            "seed": seed,
        }

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.

        Args:
            text (str): The text to count tokens in.

        Returns:
            int: Number of tokens.
        """
        return len(self.tokenizer.encode(text))

    def generate_good_samples(self, prompt: str, n_samples: int = 1, max_length: int = 1024) -> List[Tuple[str, Dict]]:
        """
        Generate multiple coherent (good) samples from a single prompt in one call.

        Args:
            prompt (str): The input prompt.
            n_samples (int): Number of samples to generate.
            max_length (int): Maximum length of the generated text in tokens.

        Returns:
            List[Tuple[str, Dict]]: List of tuples, each containing the generated text (completion) and a dictionary of generation parameters.
        """
        gen_params = {
            "max_length": max_length,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "num_return_sequences": n_samples,
        }

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = len(inputs.input_ids[0])

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)

        samples = []
        for output in outputs:
            full_text = self.tokenizer.decode(output, skip_special_tokens=True)
            # Remove the prompt from the generated text if present
            if full_text.startswith(prompt):
                completion = full_text[len(prompt) :].strip()
            else:
                completion = full_text.strip()
            completion_tokens = self.count_tokens(completion)
            sample_params = {
                **gen_params,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            samples.append((completion, sample_params))

        return samples

    def generate_bad_samples(
        self, prompt: str, n_samples: int = 1, max_length: int = 1024
    ) -> List[Tuple[str, Dict, List[str]]]:
        """
        Generate multiple incoherent (bad) samples from a single prompt in a batched manner.

        This method first generates a batch of coherent parts using num_return_sequences, then for each coherent text
        generates a degraded continuation using individually randomized parameters.

        Args:
            prompt (str): The input prompt.
            n_samples (int): Number of samples to generate.
            max_length (int): Maximum length for the entire generated text (coherent + degraded).

        Returns:
            List[Tuple[str, Dict, List[str]]]: List of tuples, each containing the full generated text, a dictionary of generation parameters, and a list of degradation methods applied.
        """

        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = len(inputs.input_ids[0])

        # Define generation parameters
        coherent_fraction = random.uniform(0.2, 0.4)  # Keep 20–40% coherent
        gen_params = {
            "max_length": max_length,
            "temperature": 1.0,  # Base temp; logits processor drives chaos
            "top_p": 1.0,  # Full distribution sampling
            "do_sample": True,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "num_return_sequences": n_samples,
            "logits_processor": [
                DegradingLogitsProcessor(
                    max_length=max_length,
                    prompt_length=prompt_tokens,
                    degradation_factor=random.uniform(0.5, 0.9),
                    top_k=random.randint(10000, 50000),  # Randomize top-K range
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
            text = self.tokenizer.decode(output, skip_special_tokens=True).strip()
            completion_tokens = self.count_tokens(text) - prompt_tokens
            params = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
            }
            degradation_methods = [
                f"coherent_fraction_{coherent_fraction:.2f}",
                "logits_degradation_repetition",
                "logits_degradation_noise",
                "logits_degradation_symbols_extended",
            ]

            bad_samples.append((text, params, degradation_methods))

        return bad_samples

    # def generate_dataset_from_file(
    #     self,
    #     file_path: str,
    #     prompt_column: str = "prompt",
    #     n_samples_per_prompt: int = 5,
    #     max_length: int = 1024,
    #     file_format: str = "csv",
    # ) -> pd.DataFrame:
    #     """
    #     Generate a dataset by reading prompts from a file.

    #     Args:
    #         file_path (str): Path to the file containing prompts.
    #         prompt_column (str): Column name in the file containing the prompt.
    #         n_samples_per_prompt (int): Number of samples to generate per prompt.
    #         max_length (int): Maximum length of generated text in tokens.
    #         file_format (str): File format (csv, parquet, or json).

    #     Returns:
    #         pd.DataFrame: DataFrame containing the generated samples.
    #     """
    #     if file_format.lower() == "csv":
    #         df = pd.read_csv(file_path)
    #     elif file_format.lower() == "parquet":
    #         df = pd.read_parquet(file_path)
    #     elif file_format.lower() == "json":
    #         df = pd.read_json(file_path)
    #     else:
    #         error_msg = f"Unsupported file format: {file_format}"
    #         logger.error(error_msg)
    #         raise ValueError(error_msg)

    #     prompts = df[prompt_column].tolist()
    #     logger.info(f"Loaded {len(prompts)} prompts from {file_path}")
    #     return self.generate_dataset(prompts, n_samples_per_prompt, max_length)

    def generate_dataset(self, prompts: List[str], n_samples_per_prompt: int = 5, max_length: int = 1024) -> pd.DataFrame:
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

        for prompt in tqdm(prompts, desc="Generating samples from prompts"):
            try:
                # Generate good samples in one batch
                # for attempt in range(self.max_attempts):
                #     try:
                #         good_samples = self.generate_good_samples(prompt, n_samples_per_prompt, max_length)
                #         if any(good_text.strip() for good_text, _ in good_samples):
                #             break
                #     except Exception as e:
                #         if attempt == self.max_attempts - 1:
                #             logger.warning(f"Failed to generate good samples after {self.max_attempts} attempts: {e}")
                #             good_samples = [
                #                 (
                #                     f"[Generation failed for prompt: {prompt[:30]}...]",
                #                     {
                #                         "prompt_tokens": self.count_tokens(prompt),
                #                         "completion_tokens": 0,
                #                         "total_tokens": self.count_tokens(prompt),
                #                     },
                #                 )
                #             ]
                #         continue

                # for good_text, good_params in good_samples:
                #     good_sample = GeneratedSample(
                #         prompt=prompt,
                #         text=good_text,
                #         is_coherent=True,
                #         model_name=self.model_name,
                #         prompt_tokens=good_params.get("prompt_tokens", 0),
                #         completion_tokens=good_params.get("completion_tokens", 0),
                #         total_tokens=good_params.get("total_tokens", 0),
                #     )
                #     all_samples.append(good_sample.model_dump())

                # Generate bad samples in one batch
                for attempt in range(self.max_attempts):
                    try:
                        bad_samples = self.generate_bad_samples(prompt, n_samples_per_prompt, max_length)
                        if any(bad_text.strip() for bad_text, _, _ in bad_samples):
                            break
                    except Exception as e:
                        if attempt == self.max_attempts - 1:
                            logger.warning(f"Failed to generate bad samples after {self.max_attempts} attempts: {e}")
                            bad_samples = [
                                (
                                    f"[Bad generation failed: {prompt[:30]}...]",
                                    {
                                        "prompt_tokens": self.count_tokens(prompt),
                                        "completion_tokens": 0,
                                        "total_tokens": self.count_tokens(prompt),
                                    },
                                    ["generation_failed"],
                                )
                            ]
                        continue

                for bad_text, bad_params, degradation_methods in bad_samples:
                    bad_sample = GeneratedSample(
                        prompt=prompt,
                        text=bad_text,
                        is_coherent=False,
                        model_name=self.model_name,
                        prompt_tokens=bad_params.get("prompt_tokens", 0),
                        completion_tokens=bad_params.get("completion_tokens", 0),
                        total_tokens=bad_params.get("total_tokens", 0),
                        degradation_methods=degradation_methods,
                    )
                    all_samples.append(bad_sample.model_dump())

            except Exception as e:
                logger.error(f"Error generating samples for prompt '{prompt[:50]}...': {e}")
                continue

        df = pd.DataFrame(all_samples)
        self.save_dataset(df)
        return df

    def save_dataset(self, df: pd.DataFrame, train_ratio: float = 0.8) -> None:
        """
        Save the generated dataset along with metadata and statistics.

        Args:
            df (pd.DataFrame): DataFrame containing the generated samples.
            train_ratio (float): Ratio of data to use for training vs validation.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        # Shuffle the DataFrame
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split into train and validation sets
        train_size = int(len(df) * train_ratio)
        train_df = df[:train_size]
        val_df = df[train_size:]

        # Save datasets
        train_file = os.path.join(run_dir, "coherence_train.parquet")
        val_file = os.path.join(run_dir, "coherence_val.parquet")

        train_df.to_parquet(train_file, index=False)
        val_df.to_parquet(val_file, index=False)

        # Calculate statistics
        train_stats = {
            "coherent": int(train_df["is_coherent"].sum()),
            "incoherent": int(len(train_df) - train_df["is_coherent"].sum()),
            "total_samples": len(train_df),
            "avg_prompt_tokens": float(train_df["prompt_tokens"].mean()),
            "avg_completion_tokens": float(train_df["completion_tokens"].mean()),
            "avg_total_tokens": float(train_df["total_tokens"].mean()),
            "max_tokens": int(train_df["total_tokens"].max()),
        }

        val_stats = {
            "coherent": int(val_df["is_coherent"].sum()),
            "incoherent": int(len(val_df) - val_df["is_coherent"].sum()),
            "total_samples": len(val_df),
            "avg_prompt_tokens": float(val_df["prompt_tokens"].mean()),
            "avg_completion_tokens": float(val_df["completion_tokens"].mean()),
            "avg_total_tokens": float(val_df["total_tokens"].mean()),
            "max_tokens": int(val_df["total_tokens"].max()),
        }

        # Count degradation methods used
        degradation_counts = {}
        for methods in df[~df["is_coherent"]]["degradation_methods"]:
            for method in methods:
                degradation_counts[method] = degradation_counts.get(method, 0) + 1

        metadata = GenerationMetadata(
            model_name=self.model_name,
            timestamp=timestamp,
            device=self.device,
            seed=self.run_metadata["seed"],
            train_ratio=train_ratio,
            train_statistics=train_stats,
            val_statistics=val_stats,
            degradation_method_counts=degradation_counts,
        )

        with open(os.path.join(run_dir, "metadata.json"), "w") as f:
            f.write(metadata.model_dump_json(indent=2))

        logger.info(f"Saved dataset to {run_dir}:")
        logger.info(f"  Training set: {train_stats['coherent']} coherent, {train_stats['incoherent']} incoherent")
        logger.info(f"  Validation set: {val_stats['coherent']} coherent, {val_stats['incoherent']} incoherent")
        logger.info(f"  Average tokens per sample: {train_stats['avg_total_tokens']:.1f}")
        logger.info(f"  Metadata and statistics saved to {os.path.join(run_dir, 'metadata.json')}")


# Example usage
if __name__ == "__main__":

    generator = SyntheticDataGenerator(model_name="Qwen/Qwen2.5-0.5B-Instruct", output_dir="./test_data/coherence_samples")

    # Example 1: Generate from a list of prompts
    prompts = [
        "Explain how photosynthesis works.",
        "Write a short story about a detective solving a mystery.",
        "Describe the process of making sourdough bread.",
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
