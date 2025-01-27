"""OpenAI generator class to call OpenAI like APIs to generate answers"""

import logging
import os
import random
import time
from typing import Dict, List, Optional

import tiktoken
from openai import OpenAI
from openai.types.chat import ChatCompletion

from rl4llm.constants import DEFAULT_FAILED_RESPONSE
from rl4llm.types import ChatTurn, EnvAction, TokenUsage

logger = logging.getLogger()


class OpenAIGenerator:
    def __init__(self, model: str, delay: float = 5.0, seed: int = 43, max_retries: int = 5, base_retry_delay: float = 3.0):
        assert model
        assert delay >= 0

        api_key = os.environ.get('OPENAI_API_KEY')
        base_url = os.environ.get('OPENAI_BASE_URL', None)

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        # self.tokenizer: tiktoken.Encoding = tiktoken.encoding_for_model(model_name=model)

        self.model = model
        self.delay = delay
        self.seed = seed
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.last_call_time = None

    def _maybe_delay(self):
        """Enforce delay between API calls to respect rate limits."""
        current_time = time.time()
        if self.last_call_time is not None:
            time_since_last_call = current_time - self.last_call_time
            remaining_delay = self.delay - time_since_last_call
            if remaining_delay > 0:
                time.sleep(remaining_delay)

    def _validate_completion(self, completion: ChatCompletion, min_length: int = 10) -> bool:
        """Validate if the completion meets our requirements."""
        try:
            if not completion or not completion.choices:
                return False

            choice = completion.choices[0]
            if not choice.message or not choice.message.content:
                return False

            content = choice.message.content.strip()
            if len(content) < min_length:
                return False

            if not completion.usage or not all(
                [completion.usage.prompt_tokens, completion.usage.completion_tokens, completion.usage.total_tokens]
            ):
                return False

            return True
        except Exception:
            return False

    def generate_with_retry(
        self,
        messages: List[Dict],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        n: int = 1,
        validation_retries: int = 3,
    ) -> ChatCompletion:
        """Enhanced generate method with validation and multiple retry strategies."""
        total_attempts = 0
        max_total_attempts = self.max_retries * (validation_retries + 1)
        last_exception = None

        while total_attempts < max_total_attempts:
            try:
                # Always enforce the delay before making an API call
                self._maybe_delay()

                # Gradually increase temperature if we're having repeated failures
                adjusted_temperature = min(temperature + (total_attempts * 0.1), 1.0)

                completion = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    seed=self.seed,
                    temperature=adjusted_temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    n=n,
                    stream=False,
                )

                # Update last call time immediately after successful API call
                self.last_call_time = time.time()

                # Validate the completion
                if self._validate_completion(completion):
                    return completion

                # If validation fails, try again with different parameters
                total_attempts += 1
                retry_delay = self.base_retry_delay * (1.5 ** (total_attempts % self.max_retries))
                time.sleep(retry_delay)

            except Exception as e:
                last_exception = e
                total_attempts += 1

                # Log the error for monitoring
                logging.warning(f"Attempt {total_attempts}/{max_total_attempts} failed: {str(e)}")

                # More aggressive exponential backoff for API errors
                retry_delay = self.base_retry_delay * (2 ** (total_attempts % self.max_retries))
                time.sleep(retry_delay)

                continue

        # If we've exhausted all retries, raise a detailed exception
        raise Exception(
            f"Failed to generate valid completion after {total_attempts} attempts. " f"Last error: {str(last_exception)}"
        )

    def generate_actions_for_rl(
        self,
        batch_states: List[List[ChatTurn]],
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        top_k: Optional[int] = 0,  # kept for compatibility
        do_sample: Optional[bool] = True,
        exploring_steps: Optional[int] = 0,
    ) -> List[EnvAction]:
        """Enhanced action generation with robust error handling and validation."""
        results = []

        for states in batch_states:
            batch_messages = [
                (
                    {'role': t.role, 'content': t.content}
                    if isinstance(t, ChatTurn)
                    else {'role': t['role'], 'content': t['content']}
                )
                for t in states
            ]

            try:
                completion = self.generate_with_retry(
                    messages=batch_messages, max_new_tokens=max_new_tokens, temperature=temperature, top_p=top_p, n=1
                )

                choice_msg = completion.choices[0].message
                usage = completion.usage

                completion_text = choice_msg.content
                # DeepSeek R1 has reasoning content in the choice message
                if hasattr(choice_msg, 'reasoning_content') and choice_msg.reasoning_content:
                    completion_text = f"<think>{choice_msg.reasoning_content}</think><answer>{completion_text}</answer>"

                action = EnvAction(
                    text=completion_text,
                    exploring_steps=exploring_steps,
                    temperature=temperature,
                    usage=TokenUsage(
                        prompt_tokens=usage.prompt_tokens,
                        completion_tokens=usage.completion_tokens,
                        total_tokens=usage.total_tokens,
                    ),
                )
                results.append(action)

            except Exception as e:
                logging.error(f"Failed to generate action for state: {str(e)}")
                # Create a minimal valid EnvAction as fallback
                fallback_action = EnvAction(
                    text=DEFAULT_FAILED_RESPONSE,
                    exploring_steps=exploring_steps,
                    temperature=temperature,
                    usage=TokenUsage(
                        prompt_tokens=0,
                        completion_tokens=0,
                        total_tokens=0,
                    ),
                )
                results.append(fallback_action)

        return results
