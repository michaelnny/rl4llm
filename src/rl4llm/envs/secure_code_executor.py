"""Gemini Pro 2.5 generated solution for running python code
in a 'relative-secure' way without accessing to Docker"""

import base64  # For encoding images
import contextlib
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
import uuid
from typing import Dict, Optional, Tuple, Union

# --- Configuration ---
ALLOWED_MODULES_WHITELIST = {
    'numpy': 'np',
    'pandas': 'pd',
    'matplotlib': 'matplotlib',  # For backend setting
    'matplotlib.pyplot': 'plt',
    'seaborn': 'sns',
    'sklearn': 'sklearn',
    'math': 'math',
    'datetime': 'datetime',
    'io': 'io',  # For StringIO, BytesIO if needed by plots
    # "os": "os", # DANGEROUS. Avoid if possible.
    # If absolutely needed for specific read-only tasks,
    # consider providing wrapped, path-checked functions.
}

# Define a base workspace directory. Each execution will get a unique subdirectory.
# Ensure the user running this script has R/W access ONLY here for code execution purposes.
DEFAULT_WORKSPACE_BASE_DIR = os.path.join(
    tempfile.gettempdir(), 'llm_ds_workspaces'
)
os.makedirs(DEFAULT_WORKSPACE_BASE_DIR, exist_ok=True)

# Resource limits (OS-dependent, primarily for Unix-like systems)
# Tuple: (soft_limit, hard_limit)
# RLIMIT_CPU: CPU time in seconds
# RLIMIT_AS: Address space (virtual memory) in bytes (approx)
# RLIMIT_NPROC: Number of processes
# RLIMIT_NOFILE: Number of open file descriptors
DEFAULT_RESOURCE_LIMITS = {
    'RLIMIT_CPU': (15, 20),  # Soft limit 15s, Hard limit 20s
    'RLIMIT_AS': (512 * 1024 * 1024, 1024 * 1024 * 1024),  # 512MB / 1GB
    # "RLIMIT_NPROC": (5, 5), # Be careful, some libraries might use threads/subprocesses
    # "RLIMIT_NOFILE": (100, 100),
}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- Helper to create the sandboxed execution script ---
def _create_sandbox_script_content(
    user_code_string: str,
    execution_workspace_dir: str,
    result_file_path: str,
    allowed_modules: Dict[str, str],
    resource_limits: Dict[str, Tuple[int, int]],
) -> str:
    """
    Creates the Python script content that will be run in the subprocess.
    This script sets up the environment, imports allowed modules, and executes the user code.
    """
    imports_code_lines = []
    for module_name, alias in allowed_modules.items():
        imports_code_lines.append(f"import {module_name} as {alias}")
    imports_code = '\n'.join(imports_code_lines)

    matplotlib_config = """
try:
    import matplotlib
    matplotlib.use('Agg') # Use a non-interactive backend for plots
    import matplotlib.pyplot as plt
except ImportError:
    matplotlib = None
    plt = None
    print("Warning: Matplotlib not available or 'Agg' backend failed.", file=sys.stderr)
"""

    resource_limit_setup_code = ''
    if sys.platform != 'win32':  # 'resource' module is not available on Windows
        resource_limit_setup_code = f"""
import resource
resource_limits_map = {{
    'RLIMIT_CPU': resource.RLIMIT_CPU,
    'RLIMIT_AS': resource.RLIMIT_AS,
    # 'RLIMIT_NPROC': resource.RLIMIT_NPROC, # Uncomment if using
    # 'RLIMIT_NOFILE': resource.RLIMIT_NOFILE, # Uncomment if using
}}
resource_limits_to_set = {json.dumps(resource_limits)}

for res_name, (soft, hard) in resource_limits_to_set.items():
    res_type = resource_limits_map.get(res_name)
    if res_type is not None:
        try:
            resource.setrlimit(res_type, (soft, hard))
        except Exception as e:
            print(f"Warning: Could not set resource limit {{res_name}}: {{e}}", file=sys.stderr)
"""
    else:
        resource_limit_setup_code = "print('Info: Resource limits not applicable on Windows via `resource` module.', file=sys.stderr)"

    script_content = f"""
import sys
import os
import traceback
import io
import contextlib
import json
import base64

# --- Resource Limits (OS-dependent) ---
try:
    {resource_limit_setup_code}
except ImportError:
    print("Warning: 'resource' module not available. Cannot set resource limits.", file=sys.stderr)
except Exception as e_res:
    print(f"Error setting resource limits: {{e_res}}", file=sys.stderr)


# --- Workspace Setup ---
# The script is already running with execution_workspace_dir as CWD.
# execution_workspace = r"{execution_workspace_dir}" # Already CWD
# os.makedirs(execution_workspace, exist_ok=True) # Should exist
# os.chdir(execution_workspace) # Already CWD

# --- Allowed Module Imports ---
{imports_code}
{matplotlib_config} # Configure matplotlib after its import

# --- Output Buffers and Result Structure ---
stdout_buffer = io.StringIO()
stderr_buffer = io.StringIO()
execution_result = {{
    "stdout": "",
    "stderr": "",
    "error_type": "",
    "error_message": "",
    "traceback": "",
    "data": None # For plots or other structured data (e.g., base64 image)
}}

# --- Globals for exec ---
# Populate globals with the aliased modules and other safe items
exec_globals = {{}}
allowed_module_map = {json.dumps(allowed_modules)}
for module_name_key, alias_val in allowed_module_map.items():
    try:
        # This relies on the imports above being successful
        exec_globals[alias_val] = eval(alias_val)
    except NameError:
        exec_globals[alias_val] = None # Module import might have failed

# Limited safe builtins
safe_builtins = {{
    'print': print, 'len': len, 'range': range, 'list': list, 'dict': dict, 'tuple': tuple,
    'str': str, 'int': int, 'float': float, 'bool': bool, 'True': True, 'False': False, 'None': None,
    'abs': abs, 'round': round, 'min': min, 'max': max, 'sum': sum, 'sorted': sorted,
    'isinstance': isinstance, 'hasattr': hasattr, 'getattr': getattr, 'setattr': setattr,
    'Exception': Exception, 'ValueError': ValueError, 'TypeError': TypeError, 'KeyError': KeyError,
    'IndexError': IndexError, 'AttributeError': AttributeError,
    'BytesIO': io.BytesIO, 'StringIO': io.StringIO, # For in-memory streams
}}
exec_globals['__builtins__'] = safe_builtins

# Helper for saving plots
def _capture_plot_as_base64(fig=None, format='png'):
    if plt is None or matplotlib is None:
        print("Matplotlib (plt) is not available for plotting.", file=sys.stderr)
        return None

    target_fig = fig if fig else plt.gcf()
    # Check if the figure has any axes, which indicates something was plotted
    if not target_fig.get_axes():
        # print("Plot is empty or not initialized. No plot captured.", file=sys.stderr)
        plt.close(target_fig) # Close even if empty to free resources
        return None

    img_buffer = io.BytesIO()
    try:
        target_fig.savefig(img_buffer, format=format, bbox_inches='tight')
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
        plt.close(target_fig) # Close the figure to free memory
        return {{ "type": f"image/{{format}}", "content": img_base64 }}
    except Exception as e:
        print(f"Error saving plot: {{e}}", file=sys.stderr)
        plt.close(target_fig) # Attempt to close even on error
    finally:
        img_buffer.close()
    return None

exec_globals['capture_plot'] = _capture_plot_as_base64

# --- Execute User Code ---
user_code_to_exec = r\"\"\"
{user_code_string}
\"\"\"

final_plot_data = None
try:
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        # The user code might call capture_plot(). We'll also try to capture any lingering plot.
        compiled_code = compile(user_code_to_exec, '<llm_generated_code>', 'exec')
        exec(compiled_code, exec_globals, {{}}) # Use exec_globals as both globals and locals

    # After execution, try to capture any plot that might have been created but not explicitly saved
    # This is a bit speculative but can be helpful.
    if plt and plt.get_fignums(): # Check if any figures are open
        # print("Attempting to capture final plot...", file=sys.stderr)
        final_plot_data = _capture_plot_as_base64()


    execution_result["stdout"] = stdout_buffer.getvalue()
    execution_result["stderr"] = stderr_buffer.getvalue() # Initial stderr
    if final_plot_data:
        execution_result["data"] = final_plot_data

except Exception as e_exec:
    execution_result["error_type"] = type(e_exec).__name__
    execution_result["error_message"] = str(e_exec)
    tb_string = traceback.format_exc()
    execution_result["traceback"] = tb_string
    # Append execution error to any stderr already captured
    execution_result["stderr"] = stderr_buffer.getvalue() + "\\n--- Traceback ---\\n" + tb_string

# --- Write result to file ---
try:
    with open(r"{result_file_path}", "w", encoding='utf-8') as f_res:
        json.dump(execution_result, f_res)
except Exception as e_json:
    # Fallback if result file writing fails
    fallback_msg = f"CRITICAL: Failed to write result to JSON file: {{e_json}}\\n"
    execution_result["stderr"] = fallback_msg + execution_result.get("stderr", "")
    # Print the whole result to stderr as a last resort
    print("---FALLBACK_RESULT_START---", file=sys.stderr)
    try:
        json.dump(execution_result, sys.stderr)
    except Exception as e_final_dump:
        print(f"Failed to dump fallback to stderr: {{e_final_dump}}", file=sys.stderr)
        print(str(execution_result), file=sys.stderr) # Raw print if json fails
    print("---FALLBACK_RESULT_END---", file=sys.stderr)
finally:
    stdout_buffer.close()
    stderr_buffer.close()
"""
    return script_content


# --- The Main Execution Function ---


def execute_python_code_securely(
    code_string: str,
    execution_workspace_dir: str,  # Caller now provides the fully prepared workspace path
    timeout_seconds: int = 30,
    allowed_modules_override: Optional[Dict[str, str]] = None,
    resource_limits_override: Optional[Dict[str, Tuple[int, int]]] = None,
) -> str:
    """
    Runs Python data science code in a more secure subprocess within a pre-prepared workspace.
    - Uses the provided execution_workspace_dir.
    - Uses a wrapper script to set resource limits and control imports.
    - Captures stdout, stderr, and structured output (like plots as base64).
    """
    script_filename = 'user_script.py'
    # The script_path is now relative to the provided execution_workspace_dir
    script_path = os.path.join(execution_workspace_dir, script_filename)
    result_filename = 'result.json'
    result_path = os.path.join(execution_workspace_dir, result_filename)

    current_allowed_modules = (
        allowed_modules_override
        if allowed_modules_override is not None
        else ALLOWED_MODULES_WHITELIST
    )
    current_resource_limits = (
        resource_limits_override
        if resource_limits_override is not None
        else DEFAULT_RESOURCE_LIMITS
    )

    sandbox_script_code = _create_sandbox_script_content(
        code_string,
        execution_workspace_dir,  # This is for the script's internal reference if needed, but CWD is key
        result_filename,  # Relative path for inside the script
        current_allowed_modules,
        current_resource_limits,
    )

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(sandbox_script_code)

    formatted_output_str = 'Execution failed: Unknown error before subprocess.'
    process_result = None
    try:
        process = subprocess.run(
            [sys.executable, script_filename],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=execution_workspace_dir,  # Critical: Run from the isolated workspace
            check=False,
        )
        process_result = process

        # ... (rest of the result parsing logic from result.json remains the same) ...
        if os.path.exists(result_path):
            with open(result_path, 'r', encoding='utf-8') as f:
                exec_result = json.load(f)
        else:
            # Fallback logic for missing result.json
            stderr_content = process.stderr if process else 'Process not run.'
            if '---FALLBACK_RESULT_START---' in stderr_content:
                try:
                    fallback_json_str = stderr_content.split(
                        '---FALLBACK_RESULT_START---'
                    )[1].split('---FALLBACK_RESULT_END---')[0]
                    exec_result = json.loads(fallback_json_str)
                except Exception as e_fallback:
                    exec_result = {
                        'stdout': process.stdout if process else '',
                        'stderr': stderr_content
                        + f"\nError: result.json not found and fallback parse failed: {e_fallback}",
                        'error_type': 'OrchestrationError',
                        'error_message': 'result.json missing',
                        'traceback': '',
                        'data': None,
                    }
            else:
                exec_result = {
                    'stdout': process.stdout if process else '',
                    'stderr': stderr_content
                    + '\nError: result.json not found.',
                    'error_type': 'OrchestrationError',
                    'error_message': 'result.json missing',
                    'traceback': '',
                    'data': None,
                }

        # --- Format the output string for the LLM ---
        result_parts = []
        stdout_val = exec_result.get('stdout', '').strip()
        if stdout_val:
            result_parts.append(f"Output:\n{stdout_val}")

        data_val = exec_result.get('data')
        if data_val:
            if (
                isinstance(data_val, dict)
                and 'type' in data_val
                and data_val['type'].startswith('image/')
            ):
                result_parts.append(
                    f"Generated Data: Image ({data_val['type']}) captured. Content not shown here."
                )
            else:
                try:
                    data_str = json.dumps(data_val, indent=2)
                    result_parts.append(f"Generated Data:\n{data_str}")
                except TypeError:
                    result_parts.append(
                        f"Generated Data: (Unserializable data object of type {type(data_val).__name__})"
                    )

        final_error_messages = []
        if exec_result.get('error_type'):
            err_msg = f"Execution Error: {exec_result['error_type']}: {exec_result['error_message']}"
            if exec_result.get('traceback'):
                short_tb = '\n'.join(
                    exec_result['traceback'].strip().splitlines()[-5:]
                )
                err_msg += f"\nTraceback (last 5 lines):\n{short_tb}"
            final_error_messages.append(err_msg)

        stderr_val = exec_result.get('stderr', '').strip()
        if stderr_val and (
            not exec_result.get('traceback')
            or exec_result.get('traceback') not in stderr_val
        ):
            final_error_messages.append(f"Stderr messages:\n{stderr_val}")

        if (
            process
            and process.returncode != 0
            and not exec_result.get('error_type')
        ):
            final_error_messages.append(
                f"Subprocess exited with error code: {process.returncode}."
            )
            if process.stderr and process.stderr.strip() not in stderr_val:
                final_error_messages.append(
                    f"Raw Subprocess Stderr:\n{process.stderr.strip()}"
                )

        if final_error_messages:
            result_parts.append(
                'Errors Encountered:\n' + '\n---\n'.join(final_error_messages)
            )

        if not result_parts:
            result_parts.append(
                'Code executed successfully with no explicit print output or generated data.'
            )
        formatted_output_str = '\n\n'.join(result_parts)

    except subprocess.TimeoutExpired:
        formatted_output_str = (
            f"Execution Failed: Timeout after {timeout_seconds} seconds."
        )
        logger.error(
            f"Timeout executing code in {execution_workspace_dir}. Code:\n{code_string[:500]}..."
        )
        if os.path.exists(result_path):
            try:
                with open(result_path, 'r', encoding='utf-8') as f_partial:
                    partial_exec_result = json.load(f_partial)
                if partial_exec_result.get('stdout'):
                    formatted_output_str += f"\nPartial Output:\n{partial_exec_result['stdout'].strip()}"
                if partial_exec_result.get('stderr'):
                    formatted_output_str += f"\nPartial Stderr/Errors:\n{partial_exec_result['stderr'].strip()}"
            except Exception as e_partial:
                logger.warning(
                    f"Could not read partial result after timeout: {e_partial}"
                )

    except Exception as e_orch:
        formatted_output_str = (
            f"Execution Orchestration Failed:\n{traceback.format_exc()}"
        )
        logger.error(
            f"Orchestration error for code:\n{code_string[:500]}...\nError: {e_orch}",
            exc_info=True,
        )
    finally:
        # The caller (MDP env) is now responsible for cleaning up the workspace
        # if it created it. If the executor created a sub-folder, it could clean that.
        # For simplicity here, we assume the caller manages the lifecycle of execution_workspace_dir.
        # If execute_python_code_securely were to create subdirectories *within*
        # execution_workspace_dir, it should clean those. But here, it just uses the provided one.
        pass

    return formatted_output_str.strip()
