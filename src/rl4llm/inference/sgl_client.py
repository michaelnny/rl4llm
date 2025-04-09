import requests
import time
import json
import logging
from typing import Optional, Dict, Any, Union, List

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SGLangClientError(Exception):
    """Custom exception for SGLang client errors."""

    pass


class SGLangClient:
    """
    A synchronous Python client for interacting with an SGLang inference server.

    Handles non-streaming generation, memory management, and weight updates.
    Retries and timeouts are configured during initialization and handled internally.
    """

    def __init__(
        self,
        host: str,
        port: int,
        api_key: Optional[str] = None,
        default_timeout: float = 120.0,  # Timeout in seconds for all requests
        retry_attempts: int = 3,  # Number of retry attempts on failure
        retry_delay: float = 2.0,  # Delay between retries in seconds
    ):
        """
        Initializes the SGLangClient.

        Args:
            host: The hostname or IP address of the SGLang server.
            port: The port number of the SGLang server.
            api_key: Optional API key for server authentication.
            default_timeout: Timeout for all HTTP requests made by this client.
            retry_attempts: Number of retry attempts for failed requests.
            retry_delay: Delay between retry attempts.
        """
        if not host.startswith(("http://", "https://")):
            self.base_url = f"http://{host}:{port}"
        else:
            self.base_url = f"{host}:{port}"  # Allow user-specified schema

        self.api_key = api_key
        self.default_timeout = default_timeout
        self.retry_attempts = max(1, retry_attempts)  # Ensure at least one attempt
        self.retry_delay = retry_delay

        self.session = requests.Session()
        if self.api_key:
            self.session.headers.update({"Authorization": f"Bearer {self.api_key}"})
        self.session.headers.update({"Content-Type": "application/json"})

        logger.info(
            f"SGLangClient initialized for server at {self.base_url} "
            f"(timeout={self.default_timeout}s, retries={self.retry_attempts}, "
            f"retry_delay={self.retry_delay}s)"
        )
        # Perform a quick health check on initialization
        try:
            self.health()
            logger.info("Successfully connected to SGLang server.")
        except SGLangClientError as e:
            logger.warning(f"Initial health check failed: {e}")

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
            SGLangClientError: If the request fails after all retries,
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
                        "status": "success",
                        "message": "Operation successful (no content returned).",
                    }

                # Attempt to parse JSON
                try:
                    return response.json()
                except json.JSONDecodeError as json_err:
                    logger.error(
                        f"Failed to decode JSON response from {method} {url}. Response text: {response.text}"
                    )
                    raise SGLangClientError(
                        f"Invalid JSON response received from server."
                    ) from json_err

            except requests.exceptions.Timeout as e:
                last_exception = e
                logger.warning(
                    f"Request timed out ({method} {url}): {e}. "
                    f"Attempt {attempt + 1}/{self.retry_attempts}."
                )
            except requests.exceptions.ConnectionError as e:
                last_exception = e
                logger.warning(
                    f"Connection error ({method} {url}): {e}. "
                    f"Attempt {attempt + 1}/{self.retry_attempts}."
                )
            except requests.exceptions.RequestException as e:
                last_exception = e
                status_code = getattr(e.response, "status_code", "N/A")
                response_text = getattr(e.response, "text", "N/A")
                logger.warning(
                    f"Request failed ({method} {url}): {e}. "
                    f"Status Code: {status_code}. Response: {response_text}. "
                    f"Attempt {attempt + 1}/{self.retry_attempts}."
                )
                # Try to parse error details if response exists and is JSON
                if e.response is not None:
                    try:
                        error_details = e.response.json()
                        logger.warning(f"Server error details: {error_details}")
                        # Optionally re-raise with server details if needed
                        # raise SGLangClientError(f"Server error: {error_details.get('error', {}).get('message', response_text)}") from e
                    except json.JSONDecodeError:
                        pass  # Response was not JSON

            # If not the last attempt, wait before retrying
            if attempt < self.retry_attempts - 1:
                logger.info(f"Retrying in {self.retry_delay} seconds...")
                time.sleep(self.retry_delay)
            else:
                error_message = (
                    f"Request failed after {self.retry_attempts} attempts "
                    f"({method} {url}). Last error: {last_exception}"
                )
                logger.error(error_message)
                raise SGLangClientError(error_message) from last_exception

        # Should be unreachable, but added for type safety
        raise SGLangClientError("Request failed unexpectedly after retries.")

    # --- Public API Methods ---

    def health(self) -> bool:
        """
        Checks the basic health of the server.
        Returns True if healthy, raises SGLangClientError otherwise.
        """
        try:
            # This endpoint returns 200 OK with no body, _request handles it
            self._request("GET", "/health")
            return True
        except SGLangClientError as e:
            logger.error(f"Health check failed: {e}")
            raise

    def generate(
        self,
        text: Optional[Union[str, List[str]]] = None,
        input_ids: Optional[Union[List[int], List[List[int]]]] = None,
        input_embeds: Optional[
            Union[List[float], List[List[float]]]
        ] = None,  # Server expects nested list usually
        image_data: Optional[List[str]] = None,  # List of base64 encoded images
        sampling_params: Optional[Dict[str, Any]] = None,
        request_id: Optional[str] = None,
        log_metrics: Optional[bool] = None,
        # Add other potential GenerateReqInput fields here if needed
        **kwargs: Any,  # Allow passthrough for future/uncommon params
    ) -> Dict[str, Any]:
        """
        Sends a non-streaming generation request to the SGLang server.

        Provide *one* of `text`, `input_ids`, or `input_embeds`.

        Args:
            text: Input prompt string or list of prompts for batching.
            input_ids: Input token IDs or list of token IDs for batching.
            input_embeds: Input embeddings or list of embeddings for batching.
            image_data: Optional list of base64 encoded image data strings.
            sampling_params: Dictionary of sampling parameters (e.g., temperature, max_new_tokens).
            request_id: Optional unique identifier for the request (maps to 'rid').
            log_metrics: Optional flag to enable/disable metric logging for this request.
            **kwargs: Additional parameters allowed by GenerateReqInput.

        Returns:
            A dictionary containing the complete generation result.

        Raises:
            SGLangClientError: If the request fails or the server returns an error.
            ValueError: If incorrect input arguments are provided.
        """
        if sum(p is not None for p in [text, input_ids, input_embeds]) != 1:
            raise ValueError(
                "Provide exactly one of 'text', 'input_ids', or 'input_embeds'."
            )

        payload = {
            "sampling_params": sampling_params or {},
            "stream": False,  # Explicitly set stream to False
            **kwargs,  # Include any extra parameters first
        }
        # Overwrite/add specific params
        if text is not None:
            payload["text"] = text
        elif input_ids is not None:
            payload["input_ids"] = input_ids
        else:  # input_embeds must be provided
            payload["input_embeds"] = input_embeds

        if image_data is not None:
            payload["image_data"] = image_data
        if request_id is not None:
            payload["rid"] = request_id
        if log_metrics is not None:
            payload["log_metrics"] = log_metrics

        result = self._request("POST", "/generate", json_data=payload)
        # _request ensures result is a dict if no exception was raised
        return result

    def release_memory(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Requests the server to release GPU memory occupation temporarily.

        Args:
            session_id: Optional session ID to release memory for. If None,
                        may release globally (server-dependent behavior).

        Returns:
            A dictionary indicating success, usually `{"status": "success", ...}`.

        Raises:
            SGLangClientError: If the request fails or the server returns an error.
        """
        payload = {}
        # The server endpoint expects ReleaseMemoryOccupationReqInput which might be empty
        # or contain session_id. Let's assume it can be empty for global release.
        if session_id is not None:
            # The actual request object definition is missing, assuming session_id is the key
            payload["session_id"] = session_id

        logger.info(f"Requesting memory release (session: {session_id})...")
        # This endpoint might return 200 OK with no body, _request handles it
        result = self._request(
            "POST", "/release_memory_occupation", json_data=payload or None
        )
        logger.info(f"Memory release request response: {result}")
        return result

    def resume_memory(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Requests the server to resume GPU memory occupation.

        Args:
            session_id: Optional session ID to resume memory for. If None,
                        may resume globally (server-dependent behavior).

        Returns:
            A dictionary indicating success, usually `{"status": "success", ...}`.

        Raises:
            SGLangClientError: If the request fails or the server returns an error.
        """
        payload = {}
        # The server endpoint expects ResumeMemoryOccupationReqInput
        if session_id is not None:
            # Assuming session_id is the key based on release_memory
            payload["session_id"] = session_id

        logger.info(f"Requesting memory resume (session: {session_id})...")
        # This endpoint might return 200 OK with no body, _request handles it
        result = self._request(
            "POST", "/resume_memory_occupation", json_data=payload or None
        )
        logger.info(f"Memory resume request response: {result}")
        return result

    def update_weights_from_disk(
        self,
        model_path: str,
        skip_tokenizer_init: bool = False,
        group_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Updates the model weights from files on the server's disk.

        Args:
            model_path: The *server-side* path to the new model weights directory.
            skip_tokenizer_init: Whether to skip tokenizer re-initialization.
            group_id: Optional group ID for coordinated updates.

        Returns:
            A dictionary containing the result status (success, message, num_paused_requests).

        Raises:
            SGLangClientError: If the request fails or the server returns an error,
                               or if the server reports failure in the response.
        """
        payload = {
            "model_path": model_path,
            "skip_tokenizer_init": skip_tokenizer_init,
        }
        if group_id is not None:
            payload["group_id"] = group_id  # Matches UpdateWeightFromDiskReqInput

        logger.info(f"Requesting weight update from disk: {model_path}...")
        result = self._request("POST", "/update_weights_from_disk", json_data=payload)
        logger.info(f"Weight update from disk response: {result}")

        # Check for success flag in response as per server implementation
        if not result.get("success"):
            raise SGLangClientError(
                f"Server reported failure updating weights from disk: {result.get('message', 'Unknown error')}"
            )
        return result

    def get_weights_by_name(self, names: List[str]) -> Dict[str, Any]:
        """
        Retrieves model parameters (weights/tensors) by their names.

        Note: The server returns the weights, likely serialized (e.g., as nested lists).
              Be mindful of potential memory usage for large tensors.

        Args:
            names: A list of parameter names to retrieve.

        Returns:
            A dictionary where keys are parameter names and values are the
            retrieved weights (likely as lists or similar JSON-serializable format).

        Raises:
            SGLangClientError: If the request fails or the server returns an error.
        """
        payload = {"names": names}  # Matches GetWeightsByNameReqInput
        logger.info(f"Requesting weights: {names}...")
        result = self._request("POST", "/get_weights_by_name", json_data=payload)
        logger.info(f"Received response for {len(names)} weights.")
        # Optional: Add checks if the result structure is as expected
        # for name in names:
        #     if name not in result:
        #         logger.warning(f"Requested weight '{name}' not found in response.")
        return result

    def init_weights_update_group(
        self, group_id: str, names: List[str]
    ) -> Dict[str, Any]:
        """
        Initializes a parameter update group on the server.

        Args:
            group_id: The unique identifier for the update group.
            names: List of parameter names included in this group.

        Returns:
            A dictionary indicating success or failure from the server.

        Raises:
            SGLangClientError: If the request fails, the server returns an error status,
                               or if the server reports failure in the response body.
        """
        payload = {
            "group_id": group_id,
            "names": names,
        }  # Matches InitWeightsUpdateGroupReqInput
        logger.info(
            f"Initializing weights update group '{group_id}' for names: {names}..."
        )
        result = self._request("POST", "/init_weights_update_group", json_data=payload)
        logger.info(f"Init weights update group response: {result}")

        if not result.get("success"):
            raise SGLangClientError(
                f"Server reported failure initializing group: {result.get('message', 'Unknown error')}"
            )
        return result

    def update_weights_from_distributed(
        self, group_id: str, tensor_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Updates model parameters from distributed sources (e.g., online training).

        *** Important ***: Assumes `tensor_data` contains JSON-serializable representations
        of tensors (e.g., nested lists) as expected by the SGLang server endpoint
        `/update_weights_from_distributed`. Sending large raw tensors this way can be
        inefficient or infeasible. Verify server expectations.

        Args:
            group_id: The identifier of the initialized update group.
            tensor_data: A dictionary where keys are parameter names (matching those
                         in the initialized group) and values are the corresponding
                         tensor data (in a server-expected JSON-serializable format).

        Returns:
            A dictionary indicating success or failure from the server.

        Raises:
            SGLangClientError: If the request fails, the server returns an error status,
                               or if the server reports failure in the response body.
        """
        logger.warning(
            "update_weights_from_distributed assumes tensor_data is JSON-serializable as expected by the server."
        )

        # The server code implies UpdateWeightsFromDistributedReqInput exists,
        # but its definition isn't shown. Assuming it takes group_id and the data.
        # The exact key for tensor_data might differ ('data' is a guess).
        payload = {"group_id": group_id, "data": tensor_data}
        logger.info(f"Requesting distributed weight update for group '{group_id}'...")
        result = self._request(
            "POST", "/update_weights_from_distributed", json_data=payload
        )
        logger.info(f"Distributed weight update response: {result}")

        if not result.get("success"):
            raise SGLangClientError(
                f"Server reported failure updating weights distributed: {result.get('message', 'Unknown error')}"
            )
        return result


# --- Example Usage ---
if __name__ == "__main__":
    # Replace with your server's host and port
    SGLANG_HOST = "127.0.0.1"
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
        print("Checking server health...")
        if client.health():
            print("Server is healthy.")

        # --- Simple Generation ---
        print("\nRunning simple generation...")
        generation_result = client.generate(
            text="The capital of France is",
            sampling_params={"max_new_tokens": 10, "temperature": 0.7},
        )
        print(f"Generation Result: {generation_result}")
        generated_text = generation_result.get("text", "")
        print(f"Generated Text: '{generated_text}'")

        # --- Memory Management (Example - uncomment to run) ---
        # print("\nTesting memory management...")
        # try:
        #     print("Releasing memory...")
        #     release_resp = client.release_memory() # Global release
        #     print(f"Release response: {release_resp}")
        #     time.sleep(2) # Give time for potential effect
        #     print("Resuming memory...")
        #     resume_resp = client.resume_memory() # Global resume
        #     print(f"Resume response: {resume_resp}")
        # except SGLangClientError as e:
        #     print(f"Memory management test failed: {e}")

        # --- Weight Update from Disk (Example - requires server setup) ---
        # print("\nTesting weight update from disk (requires server path)...")
        # SERVER_MODEL_PATH = "/path/on/server/to/new_weights" # IMPORTANT: Change this
        # try:
        #     # Ensure the path exists *on the server* before running this
        #     update_resp = client.update_weights_from_disk(model_path=SERVER_MODEL_PATH)
        #     print(f"Weight update response: {update_resp}")
        # except SGLangClientError as e:
        #     print(f"Weight update from disk failed: {e}")
        # except Exception as e:
        #      print(f"Weight update from disk failed with unexpected error: {e}")

        # --- Get Weights by Name (Example) ---
        # print("\nTesting get weights by name...")
        # try:
        #     # Replace with actual parameter names from your model
        #     param_names = ["model.layers.0.mlp.fc1.weight", "model.embed_tokens.weight"]
        #     weights = client.get_weights_by_name(names=param_names)
        #     print(f"Received weights for: {list(weights.keys())}")
        #     # Accessing weights['model.layers.0.mlp.fc1.weight'] would give the data (likely list)
        # except SGLangClientError as e:
        #     print(f"Get weights failed: {e}")
        # except Exception as e:
        #      print(f"Get weights failed with unexpected error: {e}")

        # --- Distributed Weight Update (Example - requires setup and JSON-serializable data) ---
        # print("\nTesting distributed weight update (requires group init and serializable data)...")
        # try:
        #     group_id = "my_update_group_1"
        #     param_names_to_update = ["model.layers.10.mlp.fc1.weight"] # Example param
        #     print(f"Initializing group '{group_id}'...")
        #     init_resp = client.init_weights_update_group(group_id=group_id, names=param_names_to_update)
        #     print(f"Group init response: {init_resp}")

        #     # *** CRITICAL: Prepare tensor_data in a JSON-serializable format ***
        #     # This usually means converting numpy arrays/torch tensors to nested lists.
        #     # Example: Assuming fc1.weight is a 2D tensor
        #     dummy_weight_data = [[0.1, 0.2], [0.3, 0.4]] # Replace with actual data conversion
        #     tensor_payload = {param_names_to_update[0]: dummy_weight_data}

        #     print(f"Updating weights for group '{group_id}'...")
        #     update_resp = client.update_weights_from_distributed(group_id=group_id, tensor_data=tensor_payload)
        #     print(f"Distributed update response: {update_resp}")

        # except SGLangClientError as e:
        #     print(f"Distributed weight update test failed: {e}")
        # except Exception as e:
        #      print(f"Distributed weight update test failed with unexpected error: {e}")

    except SGLangClientError as e:
        print(f"\nAn error occurred: {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
