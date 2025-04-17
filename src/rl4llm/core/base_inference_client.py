"""Implements a basic HTTP based inference client
that can call standalone inference server like inference"""

import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

import requests


class InferenceClientError(Exception):
    """Custom exception for Inference client errors."""

    pass


class InferenceClient(ABC):
    """
    Base Inference client for calling inference server using HTTP methods.
    """

    def __init__(
        self,
        host: str,
        port: int,
        api_key: Optional[str] = None,
        cohost_mode: bool = True,
        default_timeout: float = 120.0,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
    ):
        """
        Initializes the InferenceClient.

        Args:
            host: The hostname or IP address of the inference server.
            port: The port number of the inference server.
            api_key: Optional API key for server authentication.
            cohost_mode: Cohost inference engine with training models.
            default_timeout: Timeout for all HTTP requests made by this client.
            retry_attempts: Number of retry attempts for failed requests.
            retry_delay: Delay between retry attempts.
        """
        if not host.startswith(('http://', 'https://')):
            self.base_url = f"http://{host}:{port}"
        else:
            self.base_url = f"{host}:{port}"  # Allow user-specified schema

        self.api_key = api_key
        self.cohost_mode = cohost_mode
        self.default_timeout = default_timeout
        self.retry_attempts = max(1, retry_attempts)
        self.retry_delay = retry_delay

        self._release_called = False
        self._resume_called = False

        self.logger = logging.getLogger(__name__)

        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update(
                {'Authorization': f"Bearer {self.api_key}"}
            )
        self.session.headers.update({'Content-Type': 'application/json'})

        self.logger.info(
            f"InferenceClient initialized for server at {self.base_url} "
            f"(timeout={self.default_timeout}s, retries={self.retry_attempts}, "
            f"retry_delay={self.retry_delay}s)"
        )
        # Perform a quick health check on initialization
        try:
            self.health()
            self.logger.info('Successfully connected to inference server.')
        except InferenceClientError as e:
            raise RuntimeError(f"Initial health check failed: {e}")

    def _request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Internal helper method to make HTTP requests with configured retries and timeout.
        Always expects a JSON dictionary response.

        Args:
            method: HTTP method (e.g., "GET", "POST").
            endpoint: Server endpoint (e.g., "/generate").
            json_data: Optional dictionary to send as JSON payload.

        Returns:
            The parsed JSON dictionary response.

        Raises:
            InferenceClientError: If the request fails after all retries,
                               if the server returns an error status code,
                               or if the response is not valid JSON.
        """
        url = self.base_url + endpoint
        last_exception = None

        for attempt in range(self.retry_attempts):
            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    json=json_data,
                    stream=False,  # Explicitly set stream to False
                    timeout=self.default_timeout,  # Use instance default timeout
                )

                # Raise HTTPError for bad status codes (4xx or 5xx)
                response.raise_for_status()

                # Handle potential empty success responses (e.g., 200 OK with no body)
                # Treat these as success, returning a standard dict
                if response.status_code == 200 and not response.content:
                    return {
                        'status': 'success',
                        'message': 'Operation successful (no content returned).',
                    }

                # Attempt to parse JSON
                try:
                    return response.json()
                except json.JSONDecodeError as json_err:
                    self.logger.error(
                        f"Failed to decode JSON response from {method} {url}. Response text: {response.text}"
                    )
                    raise InferenceClientError(
                        'Invalid JSON response received from server.'
                    ) from json_err

            except requests.exceptions.Timeout as e:
                last_exception = e
                self.logger.warning(
                    f"Request timed out ({method} {url}): {e}. "
                    f"Attempt {attempt + 1}/{self.retry_attempts}."
                )
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                self.logger.warning(
                    f"Connection error ({method} {url}): {e}. "
                    f"Attempt {attempt + 1}/{self.retry_attempts}."
                )
            except requests.exceptions.RequestException as e:
                last_exception = e
                status_code = getattr(e.response, 'status_code', 'N/A')
                response_text = getattr(e.response, 'text', 'N/A')
                self.logger.warning(
                    f"Request failed ({method} {url}): {e}. "
                    f"Status Code: {status_code}. Response: {response_text}. "
                    f"Attempt {attempt + 1}/{self.retry_attempts}."
                )
                # Try to parse error details if response exists and is JSON
                if e.response is not None:
                    try:
                        error_details = e.response.json()
                        self.logger.warning(
                            f"Server error details: {error_details}"
                        )
                        # Optionally re-raise with server details if needed
                        # raise InferenceClientError(f"Server error: {error_details.get('error', {}).get('message', response_text)}") from e
                    except json.JSONDecodeError:
                        pass  # Response was not JSON

            # If not the last attempt, wait before retrying
            if attempt < self.retry_attempts - 1:
                self.logger.info(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
            else:
                error_message = (
                    f"Request failed after {self.retry_attempts} attempts "
                    f"({method} {url}). Last error: {last_exception}"
                )
                self.logger.error(error_message)
                raise InferenceClientError(error_message) from last_exception

        # Should be unreachable, but added for type safety
        raise InferenceClientError('Request failed unexpectedly after retries.')

    def is_cohost_mode(self) -> bool:
        """Checks are we cohost inference engine and training models on the same devices"""
        return self.cohost_mode

    @abstractmethod
    def health(self) -> bool:
        """Checks the health status of remote inference server"""
        pass

    @abstractmethod
    def generate(
        self,
        prompts: Optional[Union[str, List[str]]] = None,
        sampling_params: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Calls the remote inference server for generate completion"""
        pass

    @abstractmethod
    def release_memory(self) -> None:
        """Calls the remote inference server to release/offload GPU memory"""
        pass

    @abstractmethod
    def resume_memory(self) -> None:
        """Calls the remote inference server to resume/load GPU memory"""
        pass

    @abstractmethod
    def update_weights_from_file(self, model_path: str, **kwargs) -> None:
        """Calls the remote inference server to update weights from a checkpoint file"""
        pass
