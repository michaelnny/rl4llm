import torch
import pandas as pd
import random
import re
import os
import json
import datetime
import hashlib
import logging
from typing import Dict, List, Union, Optional, Tuple, Any
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("synthetic_data_generator.log")],
)
logger = logging.getLogger("SyntheticDataGenerator")


class GeneratedSample(BaseModel):
    """Data structure for storing generated samples with minimal metadata."""

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
        """Pydantic model configuration."""

        arbitrary_types_allowed = True

    def __init__(self, **data):
        """Initialize the model and generate a sample ID if not provided."""
        super().__init__(**data)
        if not self.sample_id:
            content_hash = hashlib.md5(f"{self.prompt}{self.text}".encode()).hexdigest()
            self.sample_id = f"{'coherent' if self.is_coherent else 'incoherent'}_{content_hash[:10]}"


class GenerationMetadata(BaseModel):
    """Metadata for a generation run."""

    model_name: str = Field(..., description="Name of the model used for generation")
    timestamp: str = Field(..., description="Timestamp of the generation run")
    device: str = Field(..., description="Device used for generation (cuda/cpu)")
    seed: int = Field(..., description="Random seed used for reproducibility")
    train_ratio: float = Field(..., description="Ratio of data used for training vs validation")
    train_statistics: Dict[str, Any] = Field(..., description="Statistics about the training dataset")
    val_statistics: Dict[str, Any] = Field(..., description="Statistics about the validation dataset")
    degradation_method_counts: Dict[str, int] = Field(..., description="Count of each degradation method used")


class SyntheticDataGenerator:
    """Generates coherent and incoherent data for training classifier models."""

    def __init__(
        self,
        model_name: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        seed: int = 42,
        output_dir: str = "./data",
        max_attempts: int = 3,
        log_level: int = logging.INFO,
    ):
        """
        Initialize the synthetic data generator.

        Args:
            model_name: Name of the model from Hugging Face to use for generation
            device: Device to run the model on ('cuda' or 'cpu')
            seed: Random seed for reproducibility
            output_dir: Directory to save generated data
            max_attempts: Maximum number of generation attempts before skipping
            log_level: Logging level (default: INFO)
        """
        self.model_name = model_name
        self.device = device
        self.output_dir = output_dir
        self.max_attempts = max_attempts

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

        # Enhanced token degradation functions
        self.degradation_functions = {
            'join': self._degrade_join,
            'repeat': self._degrade_repeat,
            'mix_case': self._degrade_mix_case,
            'insert_symbols': self._degrade_insert_symbols,
            'camel_case': self._degrade_camel_case,
            'delete_characters': self._degrade_delete_characters,
            'swap_adjacent': self._degrade_swap_adjacent,
            'word_repetition': self._degrade_word_repetition,
            'sentence_repetition': self._degrade_sentence_repetition,
        }

        # Track metadata for the generation runs
        self.run_metadata = {
            "model_name": model_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "device": device,
            "seed": seed,
        }

    def count_tokens(self, text: str) -> int:
        """
        Count the number of tokens in the text.

        Args:
            text: The text to count tokens in

        Returns:
            Number of tokens in the text
        """
        return len(self.tokenizer.encode(text))

    def generate_good_sample(self, prompt: str, max_length: int = 1024) -> Tuple[str, Dict]:
        """
        Generate a coherent sample with detailed parameters.

        Args:
            prompt: The input prompt to generate from
            max_length: Maximum length of the generated text in tokens

        Returns:
            Tuple containing:
                - The generated text
                - Dictionary of generation parameters and token counts
        """
        gen_params = {
            "max_length": max_length,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = len(inputs.input_ids[0])

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_params)

        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        completion = full_text[len(prompt) :].strip() if full_text.startswith(prompt) else full_text.strip()
        completion_tokens = self.count_tokens(completion)

        return completion, {
            **gen_params,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

    def generate_bad_sample(self, prompt: str, max_length: int = 1024) -> Tuple[str, Dict, List[str]]:
        """
        Generate an incoherent sample with detailed parameters and degradation tracking.

        Args:
            prompt: The input prompt to generate from
            max_length: Maximum length of the generated text in tokens

        Returns:
            Tuple containing:
                - The generated degraded text
                - Dictionary of generation parameters and token counts
                - List of degradation methods applied
        """
        # Select degradation techniques for this sample
        degradation_methods = []

        # Start with a coherent beginning
        coherent_ratio = random.uniform(0.2, 0.6)
        coherent_length = int(max_length * coherent_ratio)

        # Parameters for coherent part
        coherent_params = {
            "max_length": min(coherent_length + self.count_tokens(prompt), max_length),
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        prompt_tokens = len(inputs.input_ids[0])

        with torch.no_grad():
            coherent_outputs = self.model.generate(**inputs, **coherent_params)

        coherent_text = self.tokenizer.decode(coherent_outputs[0], skip_special_tokens=True)
        if coherent_text.startswith(prompt):
            coherent_text = coherent_text[len(prompt) :].strip()

        degradation_methods.append("partial_coherent")

        # Parameters for degraded part
        temp = random.uniform(1.5, 3.0)
        top_p = random.uniform(0.9, 1.0)
        rep_penalty = random.uniform(0.6, 0.9)

        degraded_params = {
            "max_length": max_length,
            "temperature": temp,
            "top_p": top_p,
            "do_sample": True,
            "repetition_penalty": rep_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "no_repeat_ngram_size": 0,
        }

        degradation_methods.append(f"high_temperature_{temp:.1f}")
        degradation_methods.append(f"repetition_penalty_{rep_penalty:.1f}")

        # Generate the degraded part
        inputs = self.tokenizer(coherent_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            degraded_outputs = self.model.generate(**inputs, **degraded_params)

        degraded_text = self.tokenizer.decode(degraded_outputs[0], skip_special_tokens=True)
        if degraded_text.startswith(coherent_text):
            degraded_text = degraded_text[len(coherent_text) :].strip()

        # Combine the coherent and degraded parts
        full_text = coherent_text + " " + degraded_text

        # Apply token-level degradation with probability
        if random.random() < 0.3:  # 30% chance
            token_degradation = self._select_token_degradations()
            full_text = self._apply_token_degradation(full_text, token_degradation)
            degradation_methods.extend(token_degradation)

        # Apply sentence-level repetition with probability
        if random.random() < 0.2:  # 20% chance
            full_text = self._degrade_sentence_repetition(full_text)
            degradation_methods.append("sentence_repetition")

        completion_tokens = self.count_tokens(full_text)

        combined_params = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        return full_text, combined_params, degradation_methods

    def _select_token_degradations(self) -> List[str]:
        """
        Select which token degradation methods to use.

        Returns:
            List of degradation method names to apply
        """
        num_methods = random.randint(1, 3)
        return random.sample(list(self.degradation_functions.keys()), num_methods)

    def _apply_token_degradation(self, text: str, degradation_types: List[str]) -> str:
        """
        Apply token-level degradation using selected methods.

        Args:
            text: The text to degrade
            degradation_types: List of degradation method names to apply

        Returns:
            Degraded text
        """
        # Split text to tokens that keep punctuation and spaces
        tokens = re.findall(r'\b\w+\b|\s+|[^\w\s]', text)
        degraded_tokens = []
        total_tokens = len(tokens)

        for i, token in enumerate(tokens):
            # Make degradation more likely in later tokens
            degradation_prob = min(0.2, (i / total_tokens) * 0.5)
            if re.match(r'\b\w+\b', token) and random.random() < degradation_prob:
                # Pick a random degradation function from selected types
                degradation_type = random.choice(degradation_types)
                token = self.degradation_functions[degradation_type](token)
            degraded_tokens.append(token)

        return "".join(degraded_tokens)

    def _degrade_join(self, token: str) -> str:
        """
        Join tokens by removing spaces (no actual change to individual token).

        Args:
            token: The token to process

        Returns:
            The processed token
        """
        return token

    def _degrade_repeat(self, token: str) -> str:
        """
        Repeat characters within a token.

        Args:
            token: The token to degrade

        Returns:
            Token with repeated characters
        """
        if len(token) > 2:
            pos = random.randint(0, len(token) - 1)
            char = token[pos]
            repetitions = random.randint(2, 5)
            token = token[:pos] + char * repetitions + token[pos + 1 :]
        return token

    def _degrade_mix_case(self, token: str) -> str:
        """
        Randomly mix uppercase and lowercase letters.

        Args:
            token: The token to degrade

        Returns:
            Token with mixed case
        """
        return ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in token)

    def _degrade_insert_symbols(self, token: str) -> str:
        """
        Insert random symbols into tokens.

        Args:
            token: The token to degrade

        Returns:
            Token with inserted symbols
        """
        symbols = '!@#$%^&*()-_=+[]{}|;:,.<>?/'
        if len(token) > 2:
            pos = random.randint(0, len(token) - 1)
            symbol = random.choice(symbols)
            token = token[:pos] + symbol + token[pos:]
        return token

    def _degrade_camel_case(self, token: str) -> str:
        """
        Convert tokens to camelCase format.

        Args:
            token: The token to degrade

        Returns:
            Token in camelCase format
        """
        if len(token) > 3:
            parts = []
            remaining = token
            while remaining and len(parts) < 3:
                split_point = random.randint(1, len(remaining)) if len(remaining) > 1 else 1
                parts.append(remaining[:split_point])
                remaining = remaining[split_point:]
            token = parts[0].lower() + ''.join(p.capitalize() for p in parts[1:])
        return token

    def _degrade_delete_characters(self, token: str) -> str:
        """
        Randomly delete characters from a token.

        Args:
            token: The token to degrade

        Returns:
            Token with characters deleted
        """
        if len(token) > 3:
            num_to_delete = random.randint(1, min(3, len(token) - 2))
            positions = random.sample(range(len(token)), num_to_delete)
            return ''.join(c for i, c in enumerate(token) if i not in positions)
        return token

    def _degrade_swap_adjacent(self, token: str) -> str:
        """
        Swap adjacent characters in a token.

        Args:
            token: The token to degrade

        Returns:
            Token with adjacent characters swapped
        """
        if len(token) > 3:
            pos = random.randint(0, len(token) - 2)
            chars = list(token)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
            return ''.join(chars)
        return token

    def _degrade_word_repetition(self, token: str) -> str:
        """
        Repeat the entire token.

        Args:
            token: The token to degrade

        Returns:
            Token repeated
        """
        if random.random() < 0.3 and len(token) > 1:
            return token + " " + token
        return token

    def _degrade_sentence_repetition(self, text: str) -> str:
        """
        Repeat sentences in the text to create unnatural repetition.

        Args:
            text: The text to degrade

        Returns:
            Text with repeated sentences
        """
        # Split the text into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)

        if len(sentences) <= 1:
            return text

        # Choose how many sentences to repeat
        num_to_repeat = min(random.randint(1, 3), len(sentences))

        # Choose which sentences to repeat
        sentences_to_repeat = random.sample(range(len(sentences)), num_to_repeat)

        # Choose where to insert the repetitions
        result = []
        for i, sentence in enumerate(sentences):
            result.append(sentence)

            # With some probability, insert a repetition after this sentence
            if i in sentences_to_repeat:
                # Choose a random sentence to repeat here
                repeat_idx = random.choice(sentences_to_repeat)
                result.append(sentences[repeat_idx])

        return " ".join(result)

    def generate_dataset_from_file(
        self,
        file_path: str,
        prompt_column: str = "prompt",
        n_samples_per_prompt: int = 5,
        max_length: int = 1024,
        file_format: str = "csv",
    ) -> pd.DataFrame:
        """
        Generate dataset from file containing prompts.

        Args:
            file_path: Path to the file containing prompts
            prompt_column: Name of the column containing prompts
            n_samples_per_prompt: Number of samples to generate per prompt
            max_length: Maximum length of generated text in tokens
            file_format: Format of the input file (csv, parquet, json)

        Returns:
            DataFrame containing the generated samples
        """
        # Load prompts from file
        if file_format.lower() == "csv":
            df = pd.read_csv(file_path)
        elif file_format.lower() == "parquet":
            df = pd.read_parquet(file_path)
        elif file_format.lower() == "json":
            df = pd.read_json(file_path)
        else:
            error_msg = f"Unsupported file format: {file_format}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        prompts = df[prompt_column].tolist()
        logger.info(f"Loaded {len(prompts)} prompts from {file_path}")
        return self.generate_dataset(prompts, n_samples_per_prompt, max_length)

    def generate_dataset(self, prompts: List[str], n_samples_per_prompt: int = 5, max_length: int = 1024) -> pd.DataFrame:
        """
        Generate a dataset of good and bad samples with improved data structure.

        Args:
            prompts: List of prompts to generate from
            n_samples_per_prompt: Number of samples to generate per prompt
            max_length: Maximum length of generated text in tokens

        Returns:
            DataFrame containing the generated samples
        """
        all_samples = []

        for prompt in tqdm(prompts, desc="Generating samples from prompts"):
            for _ in range(n_samples_per_prompt):
                # Generate good sample
                try:
                    for attempt in range(self.max_attempts):
                        try:
                            good_text, good_params = self.generate_good_sample(prompt, max_length)
                            if good_text.strip():  # Check if not empty
                                break
                        except Exception as e:
                            if attempt == self.max_attempts - 1:
                                logger.warning(f"Failed to generate good sample after {self.max_attempts} attempts: {e}")
                                good_text, good_params = f"[Generation failed for prompt: {prompt[:30]}...]", {
                                    "prompt_tokens": self.count_tokens(prompt),
                                    "completion_tokens": 0,
                                    "total_tokens": self.count_tokens(prompt),
                                }
                            continue

                    good_sample = GeneratedSample(
                        prompt=prompt,
                        text=good_text,
                        is_coherent=True,
                        model_name=self.model_name,
                        prompt_tokens=good_params.get("prompt_tokens", 0),
                        completion_tokens=good_params.get("completion_tokens", 0),
                        total_tokens=good_params.get("total_tokens", 0),
                    )
                    all_samples.append(good_sample.dict())

                    # Generate bad sample
                    for attempt in range(self.max_attempts):
                        try:
                            bad_text, bad_params, degradation_methods = self.generate_bad_sample(prompt, max_length)
                            if bad_text.strip():  # Check if not empty
                                break
                        except Exception as e:
                            if attempt == self.max_attempts - 1:
                                logger.warning(f"Failed to generate bad sample after {self.max_attempts} attempts: {e}")
                                bad_text, bad_params, degradation_methods = (
                                    f"[Bad generation failed: {prompt[:30]}...]",
                                    {
                                        "prompt_tokens": self.count_tokens(prompt),
                                        "completion_tokens": 0,
                                        "total_tokens": self.count_tokens(prompt),
                                    },
                                    ["generation_failed"],
                                )
                            continue

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
                    all_samples.append(bad_sample.dict())

                except Exception as e:
                    logger.error(f"Error generating samples for prompt '{prompt[:50]}...': {e}")
                    continue

        # Convert to DataFrame
        df = pd.DataFrame(all_samples)

        # Save the datasets
        self.save_dataset(df)

        return df

    def save_dataset(self, df: pd.DataFrame, train_ratio: float = 0.8) -> None:
        """
        Save dataset with improved organization and metadata.

        Args:
            df: DataFrame containing the generated samples
            train_ratio: Ratio of data to use for training vs validation
        """
        # Create timestamp for this run
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(self.output_dir, f"run_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        # Shuffle the dataframe
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        # Split into train and validation sets
        train_size = int(len(df) * train_ratio)
        train_df = df[:train_size]
        val_df = df[train_size:]

        # Save to parquet
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

        # Save metadata and statistics
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
            f.write(metadata.json(indent=2))

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
    dataset = generator.generate_dataset(prompts, n_samples_per_prompt=4, max_length=1024)

    print(len(dataset))

    # Example 2: Generate from a CSV file
    # generator.generate_dataset_from_file(
    #     "prompts.csv",
    #     prompt_column="question",
    #     n_samples_per_prompt=3,
    #     file_format="csv"
    # )
