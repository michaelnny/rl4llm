"""A simple HTTP based client for SGLang inference server"""

import time
from typing import Any, Dict, List, Optional, Union

from rl4llm.core.base_inference_client import (
    InferenceClient,
    InferenceClientError,
)


class SGLangClient(InferenceClient):
    """
    A synchronous Python client for interacting with an SGLang inference server.

    Handles non-streaming generation, memory management, and weight updates.
    Retries and timeouts are configured during initialization and handled internally.
    """

    def health(self) -> bool:
        """
        Checks the basic health of the server.
        Returns True if healthy, raises InferenceClientError otherwise.
        """
        try:
            # This endpoint returns 200 OK with no body, _request handles it
            self._request('GET', '/health')
            return True
        except InferenceClientError as e:
            self.logger.error(f"Health check failed: {e}")
            raise

    def generate(
        self,
        prompts: Optional[Union[str, List[str]]],
        sampling_params: Optional[Dict[str, Any]] = None,
        # Add other potential GenerateReqInput fields here if needed
        **kwargs: Any,  # Allow passthrough for future/uncommon params
    ) -> Dict[str, Any]:
        """
        Sends a non-streaming generation request to the SGLang server.

        Args:
            prompts: Input prompt string or list of prompts for batching.
            sampling_params: Dictionary of sampling parameters (e.g., temperature, max_new_tokens).
            **kwargs: Additional parameters allowed by GenerateReqInput.

        Returns:
            A dictionary containing the complete generation result.

        Raises:
            InferenceClientError: If the request fails or the server returns an error.
            ValueError: If incorrect input arguments are provided.
        """
        if prompts is None or len(prompts) == 0:
            raise ValueError("Provide exactly valid input 'text'.")

        payload = {
            'sampling_params': sampling_params or {},
            'stream': False,
            **kwargs,
        }
        # Overwrite/add specific params
        if prompts is not None:
            payload['text'] = prompts

        result = self._request('POST', '/generate', json_data=payload)
        return result

    def chat_completion(
        self,
        messages: Optional[Union[str, List[str]]],
        sampling_params: Optional[Dict[str, Any]] = None,
        # Add other potential GenerateReqInput fields here if needed
        **kwargs: Any,  # Allow passthrough for future/uncommon params
    ) -> Dict[str, Any]:
        """
        Sends a non-streaming chat-completion request to the SGLang server.

        Args:
            messages: Input chat messages.
            sampling_params: Dictionary of sampling parameters (e.g., temperature, max_new_tokens).
            **kwargs: Additional parameters allowed by GenerateReqInput.

        Returns:
            A dictionary containing the complete generation result.

        Raises:
            InferenceClientError: If the request fails or the server returns an error.
            ValueError: If incorrect input arguments are provided.
        """
        if messages is None or len(messages) == 0:
            raise ValueError("Provide valid 'messages'.")

        payload = {
            'messages': messages,
            'stream': False,
            **sampling_params,
            **kwargs,
        }

        result = self._request('POST', '/v1/chat/completion', json_data=payload)
        return result

    def release_memory(self) -> None:
        """
        Requests the server to release GPU memory occupation temporarily.

        Raises:
            InferenceClientError: If the request fails or the server returns an error.
        """

        if not self.cohost_mode:
            return

        if self._release_called:
            self.logger.warning('Already called the release_memory.')
            return

        self.logger.info('Requesting memory release ...')
        # This endpoint might return 200 OK with no body, _request handles it
        result = self._request('POST', '/release_memory_occupation', {})
        self._release_called = True
        self._resume_called = False

        self.logger.info(f"Memory release request response: {result}")
        return result

    def resume_memory(self) -> None:
        """
        Requests the server to resume GPU memory occupation.

        Raises:
            InferenceClientError: If the request fails or the server returns an error.
        """

        if not self.cohost_mode:
            return

        if self._resume_called:
            self.logger.warning('Already called the resume_memory.')
            return

        self.logger.info('Requesting memory resume ...')
        # This endpoint might return 200 OK with no body, _request handles it
        result = self._request('POST', '/resume_memory_occupation', {})
        self._resume_called = True
        self._release_called = False

        self.logger.info(f"Memory resume request response: {result}")

    def update_weights_from_file(
        self,
        model_path: str,
        skip_tokenizer_init: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Updates the model weights from files on the server's disk.

        Args:
            model_path: The *server-side* path to the new model weights directory.
            skip_tokenizer_init: Whether to skip tokenizer re-initialization.
            **kwargs: Additional parameters.

        Returns:
            A dictionary containing the result status (success, message, num_paused_requests).

        Raises:
            InferenceClientError: If the request fails or the server returns an error,
                               or if the server reports failure in the response.
        """
        payload = {
            'model_path': model_path,
            'skip_tokenizer_init': skip_tokenizer_init,
        }

        self.logger.info(f"Requesting weight update from disk: {model_path}...")
        result = self._request(
            'POST', '/update_weights_from_disk', json_data=payload
        )
        self._release_called = False
        self._resume_called = False
        self.logger.info(f"Weight update from disk response: {result}")

        # Check for success flag in response as per server implementation
        if not result.get('success'):
            raise InferenceClientError(
                f"Server reported failure updating weights from disk: {result.get('message', 'Unknown error')}"
            )
        return result


# --- Example Usage ---
if __name__ == '__main__':
    import torch
    from transformers import AutoModelForCausalLM

    MODEL_NAME = 'Qwen/Qwen2.5-0.5B'

    # Replace with your server's host and port
    SGLANG_HOST = 'localhost'
    SGLANG_PORT = 30000
    # SGLANG_API_KEY = "your_api_key_if_set" # Optional

    # Initialize the client with desired retry/timeout settings
    client = SGLangClient(
        host=SGLANG_HOST,
        port=SGLANG_PORT,
        # api_key=SGLANG_API_KEY,
        default_timeout=60.0,  # seconds
        retry_attempts=2,
        retry_delay=1.0,
    )

    try:
        # --- Health Check ---
        print('Checking server health...')
        if client.health():
            print('Server is healthy.')

        # --- Simple Generation ---
        print('\nRunning simple generation...')
        generation_result = client.generate(
            text='The capital of France is',
            sampling_params={'max_new_tokens': 10, 'temperature': 0.7},
        )
        print(f"Generation Result: {generation_result}")
        generated_text = generation_result.get('text', '')
        print(f"Generated Text: '{generated_text}'")

    except InferenceClientError as e:
        print(f"\nAn error occurred: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
