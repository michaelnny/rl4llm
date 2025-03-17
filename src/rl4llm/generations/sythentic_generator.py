import torch
import pandas as pd
import random
import re
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


class SyntheticDataGenerator:
    """Generates incoherent data for training classifier model"""

    def __init__(
        self,
        model_name="meta-llama/Llama-2-7b-chat-hf",  # Model name from HF
        device="cuda",  # Use "cuda" or "cpu"
        seed=42,  # For reproducibility
        output_dir="./data",  # Where to save generated data
    ):
        self.device = device
        self.output_dir = output_dir

        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Set seeds for reproducibility
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        print(f"Loading model {model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
        ).to(device)
        print("Model loaded successfully!")

        # If the model doesn't have a padding token, set it to the eos token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Keep the original token degradation functions
        self.degradation_functions = {
            'join': self._degrade_join,
            'repeat': self._degrade_repeat,
            'mix_case': self._degrade_mix_case,
            'insert_symbols': self._degrade_insert_symbols,
            'camel_case': self._degrade_camel_case,
        }

    def generate_good_sample(self, prompt, max_length=1024):
        """Generate a coherent sample"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt) :].strip()
        return generated_text

    def generate_bad_sample(self, prompt, max_length=1024):
        """Generate an incoherent sample with progressive degradation"""
        # Start with a coherent beginning
        coherent_ratio = random.uniform(0.2, 0.6)
        coherent_length = int(max_length * coherent_ratio)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            coherent_outputs = self.model.generate(
                **inputs,
                max_length=min(coherent_length + len(inputs.input_ids[0]), max_length),
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        coherent_text = self.tokenizer.decode(coherent_outputs[0], skip_special_tokens=True)
        if coherent_text.startswith(prompt):
            coherent_text = coherent_text[len(prompt) :].strip()

        # Generate the degraded part with higher temperature
        inputs = self.tokenizer(coherent_text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            # Use higher temperature and lower repetition penalty for natural degradation
            degraded_outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=random.uniform(1.5, 3.0),  # High temperature = more random
                top_p=random.uniform(0.9, 1.0),  # Less restrictive sampling
                do_sample=True,
                repetition_penalty=random.uniform(0.6, 0.9),  # Lower penalty = more repetition
                pad_token_id=self.tokenizer.pad_token_id,
                no_repeat_ngram_size=0,  # Disable n-gram repetition blocking
            )
        degraded_text = self.tokenizer.decode(degraded_outputs[0], skip_special_tokens=True)
        if degraded_text.startswith(coherent_text):
            degraded_text = degraded_text[len(coherent_text) :].strip()

        # Combine the coherent and degraded parts
        full_text = coherent_text + " " + degraded_text

        # Optionally apply token-level degradation
        if random.random() < 0.3:  # 30% chance
            full_text = self._apply_token_degradation(full_text)

        return full_text

    def _apply_token_degradation(self, text):
        """Apply token-level degradation"""
        # Split text to tokens that keep punctuation and spaces
        tokens = re.findall(r'\b\w+\b|\s+|[^\w\s]', text)
        degraded_tokens = []
        total_tokens = len(tokens)

        for i, token in enumerate(tokens):
            # Make degradation more likely in later tokens
            degradation_prob = min(0.2, (i / total_tokens) * 0.5)
            if re.match(r'\b\w+\b', token) and random.random() < degradation_prob:
                # Pick a random degradation function
                degradation_type = random.choice(list(self.degradation_functions.keys()))
                token = self.degradation_functions[degradation_type](token)
            degraded_tokens.append(token)

        return "".join(degraded_tokens)

    # Original token degradation functions
    def _degrade_join(self, token):
        return token

    def _degrade_repeat(self, token):
        if len(token) > 2:
            pos = random.randint(0, len(token) - 1)
            char = token[pos]
            repetitions = random.randint(2, 5)
            token = token[:pos] + char * repetitions + token[pos + 1 :]
        return token

    def _degrade_mix_case(self, token):
        return ''.join(c.upper() if random.random() > 0.5 else c.lower() for c in token)

    def _degrade_insert_symbols(self, token):
        symbols = '!@#$%^&*()-_=+[]{}|;:,.<>?/'
        if len(token) > 2:
            pos = random.randint(0, len(token) - 1)
            symbol = random.choice(symbols)
            token = token[:pos] + symbol + token[pos:]
        return token

    def _degrade_camel_case(self, token):
        if len(token) > 3:
            parts = []
            remaining = token
            while remaining and len(parts) < 3:
                split_point = random.randint(1, len(remaining)) if len(remaining) > 1 else 1
                parts.append(remaining[:split_point])
                remaining = remaining[split_point:]
            token = parts[0].lower() + ''.join(p.capitalize() for p in parts[1:])
        return token

    def generate_dataset(self, prompts, n_samples_per_prompt=5, max_length=1024):
        """Generate a dataset of good and bad samples"""
        all_data = []
        for prompt in tqdm(prompts, desc="Generating samples from prompts"):
            for _ in range(n_samples_per_prompt):
                good_sample = self.generate_good_sample(prompt, max_length)
                all_data.append({"text": good_sample, "is_coherent": 1})
                bad_sample = self.generate_bad_sample(prompt, max_length)
                all_data.append({"text": bad_sample, "is_coherent": 0})
        df = pd.DataFrame(all_data)
        self.save_to_parquet(df)
        return df

    def save_to_parquet(self, df, train_ratio=0.8):
        """Save dataset to parquet files"""
        df = df.sample(frac=1).reset_index(drop=True)
        train_size = int(len(df) * train_ratio)
        train_df = df[:train_size]
        val_df = df[train_size:]
        train_df.to_parquet(os.path.join(self.output_dir, "coherence_train.parquet"), index=False)
        val_df.to_parquet(os.path.join(self.output_dir, "coherence_val.parquet"), index=False)
        print(f"Saved {len(train_df)} training samples and {len(val_df)} validation samples")
        train_stats = {"coherent": train_df["is_coherent"].sum(), "incoherent": len(train_df) - train_df["is_coherent"].sum()}
        val_stats = {"coherent": val_df["is_coherent"].sum(), "incoherent": len(val_df) - val_df["is_coherent"].sum()}
        print(f"Training set: {train_stats['coherent']} coherent, {train_stats['incoherent']} incoherent")
        print(f"Validation set: {val_stats['coherent']} coherent, {val_stats['incoherent']} incoherent")
