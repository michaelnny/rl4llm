"""Implements MDP ENV with basic tool use for collect samples using SGLang inference server with a custom HTTP client"""

import contextlib
import io
import json
import logging
import os
import re
import shutil
import traceback
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from rl4llm.constants import LOGGER_NAME
from rl4llm.core.base_env import (
    BaseMDPEnv,
    ChatMessage,
    EnvState,
    SampleState,
)
from rl4llm.core.base_inference_client import InferenceClient

from .secure_code_executor import (
    ALLOWED_MODULES_WHITELIST,
    DEFAULT_WORKSPACE_BASE_DIR,
    execute_python_code_securely,
)

logger = logging.getLogger(LOGGER_NAME)


# --- Code Execution Tool ---
def execute_python_code(code_string: str) -> str:
    """Runs python code in non-secure way"""
    output_buffer = io.StringIO()
    error_buffer = io.StringIO()
    global_vars = {}
    local_vars = {}
    try:
        with (
            contextlib.redirect_stdout(output_buffer),
            contextlib.redirect_stderr(error_buffer),
        ):
            compiled_code = compile(code_string, '<string>', 'exec')
            exec(compiled_code, global_vars, local_vars)
        stdout = output_buffer.getvalue().strip()
        stderr = error_buffer.getvalue().strip()
        result_parts = []
        if stdout:
            result_parts.append(f"Output:\n{stdout}")
        if stderr:
            result_parts.append(f"Errors:\n{stderr}")
        if not stdout and not stderr:
            result_parts.append(
                'Code executed successfully with no explicit print output.'
            )
        return '\n'.join(result_parts)
    except Exception:
        return f"Execution Failed:\n{traceback.format_exc()}"
    finally:
        output_buffer.close()
        error_buffer.close()


ENV_TOOL_SCHEMAS = [
    {
        'type': 'function',
        'function': {
            'name': 'code_execution_tool',
            'description': "Executes Python code and returns the standard output or errors. The code MUST use 'print()' for any results to be captured.",
            'parameters': {
                'type': 'object',
                'properties': {
                    'code': {
                        'type': 'string',
                        'description': 'The Python code to execute.',
                    }
                },
                'required': ['code'],
            },
        },
    }
]


# ENV_TOOL_SCHEMAS = [
#     {
#         "type": "function",
#         "function": {
#             "name": "code_execution_tool",
#             "description": (
#                 "Executes Python data analysis code in a sandboxed environment. "
#                 "Returns standard output, errors, and information about generated plots. "
#                 "Code MUST use 'print()' for textual results. "
#                 "To generate and save a plot, ensure your plotting commands (e.g., using plt) are complete, "
#                 "then call the available `capture_plot()` function. For example: `plt.plot([1,2,3]); capture_plot()`. "
#                 f"Allowed modules (imported as): {', '.join([f'{m} (as {a})' for m, a in ALLOWED_MODULES_WHITELIST.items()])}. "
#                 "Code runs in a temporary, isolated workspace. If your task involves files (e.g., 'data.csv'), "
#                 "assume they are present in the current working directory and use relative paths. "
#                 "Execution is limited by time (approx 30s) and memory. "
#                 "No direct network access or arbitrary system commands are allowed."
#             ),
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "code": {
#                         "type": "string",
#                         "description": "The Python data analysis code to execute.",
#                     }
#                 },
#                 "required": ["code"],
#             },
#         },
#     }
# ]


class SglToolMDPEnv(BaseMDPEnv):
    """
    Simple one-step MDP Environment with basic tool use using SGLang inference server with a custom HTTP client.
    """

    def __init__(self, **kwargs):
        super().__init__(tool_schemas=ENV_TOOL_SCHEMAS, **kwargs)

        self.workspace_base_dir = kwargs.pop(
            'workspace_base_dir', DEFAULT_WORKSPACE_BASE_DIR
        )
        os.makedirs(self.workspace_base_dir, exist_ok=True)

    def _prepare_execution_workspace(self, sample_state: SampleState) -> str:
        """
        Creates a unique, temporary workspace for a code execution.
        Populates it with any files relevant to the current sample/task.
        Returns the path to the created workspace.
        """
        execution_id = uuid.uuid4().hex
        workspace_path = os.path.join(
            self.workspace_base_dir, f"exec_{execution_id}"
        )
        os.makedirs(workspace_path, exist_ok=True)

        # --- THIS IS WHERE YOU ADD LOGIC TO POPULATE THE WORKSPACE ---
        # Example: If the initial prompt for this sample_state contained file content:
        # initial_prompt = sample_state.messages[0].content # Assuming first message is user prompt
        # if "file_content_for_data.txt" in initial_prompt:
        #     # (You'd need a more robust way to extract this from the prompt)
        #     file_content = extract_file_content(initial_prompt, "data.txt")
        #     if file_content:
        #         with open(os.path.join(workspace_path, "data.txt"), "w") as f:
        #             f.write(file_content)

        # For now, let's assume no specific files are pre-populated for this example,
        # but the LLM is told to expect them if the task implies it.
        logger.debug('Created execution workspace: {workspace_path}')
        return workspace_path

    def _cleanup_execution_workspace(self, workspace_path: str):
        """Removes the execution workspace."""
        if (
            workspace_path
            and os.path.exists(workspace_path)
            and self.workspace_base_dir in workspace_path
        ):  # Safety check
            try:
                shutil.rmtree(workspace_path)
                logger.debug(f"Cleaned up workspace: {workspace_path}")
            except Exception as e:
                logger.error(
                    f"Failed to cleanup workspace {workspace_path}: {e}"
                )
        else:
            logger.warning(
                f"Skipped cleanup for invalid workspace path: {workspace_path}"
            )

    def _parse_tool_calls_from_content(
        self, assistant_content: str
    ) -> List[Dict]:
        """
        Parses <tool_call>RAW_PAYLOAD</tool_call> tags from the assistant's text content.
        Returns a list of dictionaries. Each dict is either a successfully parsed
        JSON payload for a tool call, or an error dictionary if parsing/validation
        failed for an attempt.

        Successful parse: {"name": "tool_name", "arguments": {...}}
        Error parse: {"error": "ErrorTypeName", "raw_payload": "...", "message": "Human readable error"}
        """
        if not assistant_content:
            return []

        # Regex to find <tool_call> CAPTURED_CONTENT </tool_call>
        # re.DOTALL makes . match newlines. Non-greedy match for content: (.*?)
        tool_call_raw_payloads = re.findall(
            r'<tool_call>(.*?)</tool_call>',
            assistant_content,
            re.DOTALL,
        )

        parsed_tool_attempts = []
        if not tool_call_raw_payloads:  # No <tool_call> tags found at all
            return []

        for raw_payload_str in tool_call_raw_payloads:
            json_to_parse = raw_payload_str.strip()

            # Define a standard error message structure for tool messages
            malformed_call_error_prefix = 'Error: Invalid tool call. '

            if not json_to_parse:
                error_detail = 'Payload was empty.'
                logger.warning('Empty tool call payload found.')
                parsed_tool_attempts.append(
                    {
                        'error': 'EmptyPayload',
                        'raw_payload': json_to_parse,  # which is empty or whitespace
                        'message': f"{malformed_call_error_prefix}{error_detail}",
                    }
                )
                continue

            try:
                tool_call_data = json.loads(json_to_parse)
            except json.JSONDecodeError as e:
                error_detail = f"JSONDecodeError: {e}."
                logger.warning(
                    f"Failed to parse tool call JSON: '{json_to_parse}'. Error: {e}"
                )
                parsed_tool_attempts.append(
                    {
                        'error': 'JSONDecodeError',
                        'raw_payload': json_to_parse,
                        'message': f"{malformed_call_error_prefix}{error_detail} Attempted payload: '{json_to_parse[:100]}'",
                    }
                )
                continue

            if (
                not isinstance(tool_call_data, dict)
                or 'name' not in tool_call_data
                or 'arguments' not in tool_call_data
                or not isinstance(tool_call_data['arguments'], dict)
            ):
                error_detail = 'Malformed structure (missing name/arguments or arguments not a dict).'
                logger.warning(
                    f"Malformed tool call JSON structure after parsing: '{json_to_parse}' resulted in: {tool_call_data}"
                )
                parsed_tool_attempts.append(
                    {
                        'error': 'MalformedStructure',
                        'raw_payload': json_to_parse,
                        'parsed_data': tool_call_data,  # Could be useful for debugging
                        'message': f"{malformed_call_error_prefix}{error_detail} Parsed: {str(tool_call_data)[:100]}",
                    }
                )
                continue

            # Successfully parsed and validated
            parsed_tool_attempts.append(tool_call_data)

        return parsed_tool_attempts

    def _execute_tool(
        self, tool_name: str, arguments: Dict, sample_state: SampleState
    ) -> str:
        # Standard error prefix for tool execution issues
        tool_error_prefix = 'Error: '

        if tool_name == 'code_execution_tool':
            code_to_execute = arguments.get('code')
            if not isinstance(code_to_execute, str):
                return f"{tool_error_prefix}'code' string argument missing or invalid for code_execution_tool."

            try:
                logger.debug(
                    f"Executing code tool with code:\n{code_to_execute[:200]}..."
                )
                result_str = execute_python_code(
                    code_string=code_to_execute,
                )
                # Check if the result_str itself indicates a known failure pattern from execute_python_code
                # This is where we adapt its output to our convention.
                # Be careful with this check if successful output could contain these strings.
                # A more robust execute_python_code would return a status and output separately.
                if (
                    'Execution Failed:' in result_str
                    or 'Errors:\n' in result_str
                ):
                    # Prepend our standard error prefix if it's not already an "Error: " type message
                    # This assumes result_str is the full error output from execute_python_code
                    return f"{tool_error_prefix}Code execution reported issues. Output: {result_str}"
                # If execute_python_code might return other non-prefixed errors, add checks here.

            except Exception as e:
                logger.error(
                    f"Exception during code execution call: {e}", exc_info=True
                )
                return (
                    f"{tool_error_prefix}Exception during tool execution: {e}"
                )

            # If we reach here, result_str is assumed to be successful output
            return result_str

        logger.warning(f"Attempted to call unknown tool: {tool_name}")
        return f"{tool_error_prefix}Unknown tool '{tool_name}'."

    @torch.inference_mode()
    def _run_interaction_loop(
        self,
        env_state: EnvState,
        llm: InferenceClient,
        sampling_params: Dict[str, Any],
        **kwargs: Optional[Dict[str, Any]],
    ) -> EnvState:
        """
        Performs a generation steps with tool-use for all samples using the SampleState structure.

        Args:
            env_state: The starting state containing a list of SampleState objects.
            llm: The language model inference client.
            sampling_params: Configuration for generation.
            **kwargs: Additional arguments (unused in default).

        Returns:
            EnvState: The final state after one generation step, with updated SampleStates.
        """

        active_indices_map = {
            i: i for i, s in enumerate(env_state.sample_states) if not s.done
        }

        for _interaction_turn in range(self.max_steps):
            if not active_indices_map:
                break

            prompts_for_llm = []
            current_batch_original_indices = []

            for original_idx in list(active_indices_map.keys()):
                sample_state = env_state.sample_states[original_idx]
                if sample_state.done:
                    active_indices_map.pop(original_idx, None)
                    continue

                messages_as_dicts = [
                    msg.model_dump(exclude_none=True)
                    for msg in sample_state.messages
                ]
                prompt = self.tokenizer.apply_chat_template(
                    messages_as_dicts,
                    tools=self.tool_schemas,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prompts_for_llm.append(prompt)
                current_batch_original_indices.append(original_idx)

            if not prompts_for_llm:
                break

            llm_raw_outputs = llm.generate(
                prompts=prompts_for_llm, sampling_params=sampling_params
            )

            for batch_i, llm_output_obj in enumerate(llm_raw_outputs):
                original_sample_idx = current_batch_original_indices[batch_i]
                sample_state = env_state.sample_states[original_sample_idx]
                if sample_state.done:
                    continue

                llm_response_text = ''
                if (
                    isinstance(llm_output_obj, dict)
                    and 'text' in llm_output_obj
                ):
                    llm_response_text = llm_output_obj['text'].strip()
                elif isinstance(llm_output_obj, str):
                    llm_response_text = llm_output_obj.strip()
                else:
                    logger.error(
                        f"Unexpected LLM output format: {llm_output_obj}"
                    )
                    llm_response_text = '<error receiving LLM response>'

                assistant_message = ChatMessage(
                    role='assistant', content=llm_response_text
                )
                sample_state.messages.append(assistant_message)

                # Parse tool calls (or attempts) from the assistant's content
                tool_call_attempts = self._parse_tool_calls_from_content(
                    llm_response_text
                )
                sample_state.current_step += 1

                if (
                    tool_call_attempts
                ):  # If there were any tool call attempts (valid or invalid)
                    for tool_attempt_info in tool_call_attempts:
                        tool_response_content = ''
                        if 'error' in tool_attempt_info:
                            # This was a malformed/invalid tool call attempt
                            tool_response_content = tool_attempt_info['message']
                            logger.warning(
                                f"Handling malformed tool call: {tool_response_content}"
                            )
                        else:
                            # This is a structurally valid tool call, proceed to execute
                            tool_name = tool_attempt_info.get('name')
                            tool_args = tool_attempt_info.get('arguments', {})
                            tool_response_content = self._execute_tool(
                                tool_name, tool_args, sample_state
                            )

                        tool_response_message = ChatMessage(
                            role='tool',
                            content=(
                                tool_response_content
                                if tool_response_content is not None
                                else ''
                            ),
                        )
                        sample_state.messages.append(tool_response_message)

                    if sample_state.current_step >= self.max_steps:
                        sample_state.done = True
                        active_indices_map.pop(original_sample_idx, None)
                    # If max_steps not reached, the loop continues for this sample,
                    # and the LLM will see the tool responses (or error messages from malformed calls).
                else:  # No <tool_call> tags found in the LLM response.
                    # LLM's response is taken as final answer for this interaction.
                    sample_state.done = True
                    active_indices_map.pop(original_sample_idx, None)

        for sample_state in env_state.sample_states:
            # Ensure last message is assistant if interaction ended abruptly
            if sample_state.messages[-1].role != 'assistant':
                sample_state.messages.append(
                    ChatMessage(
                        role='assistant',
                        content='<Interaction ended by step limit or error>',
                    )
                )
            if not sample_state.done:
                sample_state.done = True

        return env_state
