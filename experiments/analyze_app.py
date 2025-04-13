# flake8: noqa

import glob
import html
import json
import os
import re
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px  # Keep plotly for the Analytics tab
import streamlit as st

# --- Page Config ---
st.set_page_config(layout='wide', page_title='LLM Sample Analyzer')

# --- Apply Custom CSS (Revised for Dark Theme Compatibility & New Preview Class) ---
st.markdown(
    """
<style>
    /* Base theme variables for easier customization */
    :root {
        --content-bg-color: var(--streamlit-secondary-background-color, #f8f9fa);
        --main-text-color: var(--streamlit-text-color, #333333);
        --content-text-color: var(--streamlit-text-color, #eeeeee);
        --content-border-color: var(--streamlit-border-color, #dee2e6);
        --header-bg-color: #495057; /* A neutral dark gray */
        --header-text-color: white;
        --metadata-label-color: var(--streamlit-gray-70, #6c757d);
        --highlight-bg-color: rgba(255, 213, 0, 0.3); /* Softer yellow highlight */
        --highlight-text-color: var(--streamlit-text-color);
    }

    /* Main container for content blocks in details view */
    .content-container {
        background-color: var(--content-bg-color);
        color: var(--main-text-color);
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
        border: 1px solid var(--content-border-color); /* Use theme border */
    }

    /* Text within content blocks */
    .content-text {
        font-family: monospace;
        white-space: pre-wrap;
        word-break: break-word;
        font-size: 14px;
        line-height: 1.6;
    }

    /* Section headers in details view */
    .section-header {
        background-color: var(--header-bg-color);
        color: var(--header-text-color);
        padding: 8px 12px;
        border-radius: 3px;
        margin-bottom: 12px;
        font-weight: bold;
        font-size: 16px;
    }

    /* Metadata items */
    .metadata-item {
        margin: 4px 0;
        display: flex;
        font-size: 13px;
    }
    .metadata-label {
        font-weight: bold;
        min-width: 150px;
        color: var(--metadata-label-color); /* Use theme gray */
        padding-right: 10px;
    }

    /* Status indicators (Revised for Dark Mode) */
    .status-base {
        padding: 10px;
        border-radius: 3px;
        margin-top: 5px;
        font-weight: bold;
        border-left-width: 5px;
        border-left-style: solid;
        color: var(--main-text-color); /* Use theme text color */
        background-color: var(--content-bg-color); /* Match container background */
    }
    .status-good {
        border-left-color: #28a745; /* Bootstrap success */
    }
    .status-bad {
        border-left-color: #dc3545; /* Bootstrap danger */
    }
    .status-neutral {
        border-left-color: #ffc107; /* Bootstrap warning */
    }

    /* Search highlight */
    .search-highlight {
        background-color: var(--highlight-bg-color);
        padding: 1px 3px;
        border-radius: 2px;
        color: var(--highlight-text-color);
        font-weight: bold;
    }

    /* Style for the sample list items */
    .sample-list-item {
        border-bottom: 1px solid var(--content-border-color);
        padding: 10px 0;
        margin-bottom: 5px;
    }
    .sample-list-item:last-child {
        border-bottom: none; /* No border for the last item */
    }
    .sample-info {
        font-size: 0.9em;
        color: var(--metadata-label-color); /* Use metadata color for less emphasis */
        margin-bottom: 5px; /* Add space below info */
    }
    .sample-prompt-preview, .sample-completion-preview {
        font-size: 0.95em;
        margin-top: 5px;
        font-style: italic;
        color: var(--content-text-color);
        opacity: 0.9;
        line-height: 1.4; /* Adjust line height for preview */
        max-height: 7em; /* Limit height to roughly 5 lines */
        overflow: hidden; /* Hide overflow */
        display: -webkit-box; /* Enable flexbox */
        -webkit-line-clamp: 5; /* Limit to 5 lines */
        -webkit-box-orient: vertical; /* Vertical layout */
    }
    .preview-label {
        font-weight: bold;
        font-size: 0.9em;
        color: var(--metadata-label-color);
    }

</style>
""",
    unsafe_allow_html=True,
)


# --- Helper Functions ---
# @st.cache_data # Keep caching for initial load
# def load_data_from_path(file_path):
#     # ... (load_data_from_path function remains the same)
@st.cache_data
def load_data_from_path(file_path):
    """Load data with caching to improve performance"""
    try:
        if file_path.endswith('.jsonl'):
            records = []
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError as json_err:
                        # Use st.warning inside the cached function is okay for display
                        # but avoid heavy computation or side effects here.
                        # Consider logging instead if this becomes noisy.
                        print(  # Use print for logging within cache if needed
                            f"Warning: Skipping invalid JSON line {i + 1} in {os.path.basename(file_path)}: {line[:100]}... Error: {json_err}"
                        )
                        continue
            if records:
                return pd.DataFrame(records)
            return pd.DataFrame()
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        else:
            # Returning None or raising an error is better inside cache
            # than calling st.error directly. Handle the None case outside.
            print(f"Error: Unsupported file format: {file_path}")
            return None
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None


def highlight_text(text, search_term):
    """Highlights search term in text using HTML span, escaping the text first."""
    if not search_term or not isinstance(text, str):
        return html.escape(str(text))  # Escape non-strings or if no search term
    try:
        escaped_text = html.escape(text)
        # Use re.escape on the search term to handle special regex characters
        escaped_search_term = re.escape(search_term)
        # Use word boundaries for potentially better matching, but fall back
        highlighted = re.sub(
            r'\b(' + escaped_search_term + r')\b',
            r"<span class='search-highlight'>\1</span>",
            escaped_text,
            flags=re.IGNORECASE,
        )
        # Fallback if word boundaries didn't match (e.g., term starts/ends with punctuation)
        if highlighted == escaped_text:
            highlighted = re.sub(
                f"({escaped_search_term})",
                r"<span class='search-highlight'>\1</span>",
                escaped_text,
                flags=re.IGNORECASE,
            )
        return highlighted
    except Exception as e:
        # Use print or logging inside helpers called by cached functions
        print(f"Warning: Error highlighting text: {e}")
        return html.escape(text)  # Return escaped text on error


def truncate_middle(text, max_length=200, placeholder=' <......> '):
    """Truncates text showing the start and end, joined by a placeholder."""
    if not isinstance(text, str) or len(text) <= max_length:
        return html.escape(str(text))

    placeholder_len = len(placeholder)
    if max_length <= placeholder_len:
        return html.escape(text[:max_length])

    chars_to_keep = max_length - placeholder_len
    front_chars = (chars_to_keep + 1) // 2
    back_chars = chars_to_keep - front_chars

    if back_chars <= 0:  # Handle edge case where max_length is very small
        front_chars = max_length - placeholder_len
        if front_chars < 0:
            front_chars = 0  # Ensure non-negative
        return html.escape(text[:front_chars]) + placeholder

    start_part = html.escape(text[:front_chars].strip())
    end_part = html.escape(text[-back_chars:].strip())

    # Only add placeholder if text was actually truncated
    if len(text) > max_length:
        return f"{start_part}{placeholder}{end_part}"
    else:
        return html.escape(text)


# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'data_loaded': False,
        'selected_row_index': None,
        'show_details': False,
        # 'filtered_df': None, # We will get this from the cached function
        'df': None,  # The original loaded dataframe
        'current_page': 1,
        'last_files': [],
        'data_dir': os.getcwd(),
        'specific_files_input': '',
        'selected_files_cache': [],
        'selected_specific_files_cache': [],
        # Filter states - initialize properly
        'filter_sources': [],
        'filter_step_range': None,
        'filter_steps_select': [],
        'filter_reward_range': None,
        'search_term': '',
        'search_prompt': True,
        'search_completion': True,
        'rows_per_page': 10,
        # 'filters_changed_flag': False, # No longer needed
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()

# --- Command Line Argument Processing ---
# (No changes needed here)
if len(sys.argv) > 1 and 'data_dir_cli_set' not in st.session_state:
    for i in range(1, len(sys.argv)):
        if sys.argv[i].startswith('--data_dir='):
            data_dir_arg = sys.argv[i].split('=')[1]
            if os.path.isdir(data_dir_arg):
                st.session_state.data_dir = data_dir_arg
                st.session_state.data_dir_cli_set = True
            else:
                st.warning(
                    f"Command line directory not found: {data_dir_arg}",
                    icon='⚠️',
                )
            break


# --- Sidebar: File Selection ---
# (No changes needed here, logic seems fine)
st.sidebar.title('LLM Sample Analyzer')
data_source = st.sidebar.radio(
    'Data Source', ['Directory', 'Specific Files'], key='data_source_radio'
)
selected_files = []
data_dir_value = st.session_state.data_dir

if data_source == 'Directory':
    data_dir_input = st.sidebar.text_input(
        'Directory path', value=data_dir_value, key='data_dir_input_field'
    )
    if data_dir_input != st.session_state.data_dir:  # Check if path changed
        if os.path.isdir(data_dir_input):
            st.session_state.data_dir = data_dir_input
            st.session_state.selected_files_cache = (
                []
            )  # Reset cache on dir change
            st.rerun()  # Rerun to update file list
        else:
            st.sidebar.error(f"Directory not found: '{data_dir_input}'")

    data_dir = st.session_state.data_dir  # Use the potentially updated path

    if data_dir and os.path.isdir(data_dir):
        try:
            all_files = sorted(
                glob.glob(os.path.join(data_dir, '*.jsonl'))
                + glob.glob(os.path.join(data_dir, '*.parquet'))
            )
            if all_files:
                # Ensure defaults are valid files currently present
                valid_defaults = [
                    f
                    for f in st.session_state.selected_files_cache
                    if f in all_files
                ]
                # If no cached selections are valid, default to first 5 or all
                if not valid_defaults and all_files:
                    valid_defaults = all_files[: min(5, len(all_files))]

                selected_files_widget = st.sidebar.multiselect(
                    'Select files',
                    all_files,
                    default=valid_defaults,
                    key='file_multiselect_dir',
                )
                # Update cache only if selection changes
                if set(selected_files_widget) != set(  # Use set for comparison
                    st.session_state.selected_files_cache
                ):
                    st.session_state.selected_files_cache = (
                        selected_files_widget
                    )
                    # Don't rerun here, let data loading handle it below
                selected_files = (
                    st.session_state.selected_files_cache
                )  # Use cached value

            else:
                st.sidebar.info(
                    f"No JSONL or Parquet files found in '{os.path.basename(data_dir)}'."
                )
        except Exception as e:
            st.sidebar.error(f"Error accessing directory: {e}")
    elif data_dir:
        # This case handles when the initial default path is invalid
        if not os.path.isdir(data_dir):
            st.sidebar.error(f"Directory not found: '{data_dir}'")

else:  # Specific Files
    file_paths_input = st.sidebar.text_area(
        'Enter file paths (one per line)',
        value=st.session_state.specific_files_input,
        key='specific_files_text_area',
    )
    if file_paths_input != st.session_state.specific_files_input:
        st.session_state.specific_files_input = file_paths_input
        st.session_state.selected_specific_files_cache = []  # Reset cache
        st.rerun()  # Rerun to update file list

    if st.session_state.specific_files_input:
        paths_list = [
            p.strip()
            for p in st.session_state.specific_files_input.split('\n')
            if p.strip()
        ]
        valid_paths = [
            p
            for p in paths_list
            if os.path.isfile(p)
            and (p.endswith('.jsonl') or p.endswith('.parquet'))
        ]
        invalid_paths = [p for p in paths_list if p not in valid_paths]
        if invalid_paths:
            st.sidebar.warning(
                f"Invalid/missing files ignored: {', '.join(invalid_paths)}",
                icon='⚠️',
            )
        if valid_paths:
            # Ensure defaults are valid files currently present
            valid_defaults = [
                f
                for f in st.session_state.selected_specific_files_cache
                if f in valid_paths
            ]
            # If no cached selections are valid, default to all valid paths
            if not valid_defaults:
                valid_defaults = valid_paths

            selected_files_widget = st.sidebar.multiselect(
                'Confirm files',
                valid_paths,
                default=valid_defaults,
                key='file_multiselect_specific',
            )
            # Update cache only if selection changes
            if set(selected_files_widget) != set(  # Use set for comparison
                st.session_state.selected_specific_files_cache
            ):
                st.session_state.selected_specific_files_cache = (
                    selected_files_widget
                )
                # Don't rerun here
            selected_files = (
                st.session_state.selected_specific_files_cache
            )  # Use cached value

        elif not invalid_paths:
            st.sidebar.info('Enter valid file paths above.')


# --- Data Loading ---
# Load data if selected files change OR if data isn't loaded yet but files are selected
if selected_files and (
    not st.session_state.data_loaded
    or set(st.session_state.last_files)
    != set(selected_files)  # Use set for order-independent comparison
):
    with st.spinner('Loading data...'):
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        loaded_dataframes = []
        total_files = len(selected_files)
        load_errors = False

        for i, file_path in enumerate(selected_files):
            base_name = os.path.basename(file_path)
            status_text.text(f"Loading {base_name} ({i + 1}/{total_files})...")
            df = load_data_from_path(file_path)  # Uses cached function

            # Handle potential errors from load_data_from_path
            if df is None:
                st.warning(
                    f"Failed to load or parse {base_name}. Skipping.", icon='⚠️'
                )
                load_errors = True
                progress_bar.progress((i + 1) / total_files)
                continue  # Skip to next file

            if not df.empty:
                df['source_file'] = base_name
                # Define expected columns and add if missing
                expected_cols = {
                    'step': pd.NA,
                    'prompt_text': '',
                    'completion_text': '',
                    'accuracy_reward': pd.NA,
                    'prompt_length': pd.NA,
                    'completion_length': pd.NA,
                    'ground_truth': '',
                }
                for col, default_val in expected_cols.items():
                    if col not in df.columns:
                        df[col] = default_val

                # Calculate lengths if missing and text exists
                # Ensure text columns exist before trying to calculate length
                if 'prompt_text' in df.columns and (
                    'prompt_length' not in df.columns
                    or df['prompt_length'].isnull().any()
                ):
                    df['prompt_length'] = (
                        df['prompt_text'].astype(str).apply(len)
                    )
                if 'completion_text' in df.columns and (
                    'completion_length' not in df.columns
                    or df['completion_length'].isnull().any()
                ):
                    df['completion_length'] = (
                        df['completion_text'].astype(str).apply(len)
                    )

                loaded_dataframes.append(df)
            progress_bar.progress((i + 1) / total_files)

        status_text.text('Combining data...')
        if loaded_dataframes:
            try:
                # Combine all loaded dataframes
                combined_df = pd.concat(loaded_dataframes, ignore_index=True)

                # Attempt type conversions after concatenation
                col_types = {
                    'step': 'Int64',
                    'accuracy_reward': 'float64',
                    'prompt_length': 'Int64',
                    'completion_length': 'Int64',
                }
                for col, dtype in col_types.items():
                    if col in combined_df.columns:
                        try:
                            # Convert to numeric first, coercing errors
                            numeric_col = pd.to_numeric(
                                combined_df[col], errors='coerce'
                            )
                            # Then convert to the target type (Int64 handles NA)
                            if dtype == 'Int64':
                                combined_df[col] = numeric_col.astype(dtype)
                            elif dtype == 'float64':
                                combined_df[col] = numeric_col.astype(dtype)
                            # Add other types if needed
                        except Exception as type_e:
                            st.warning(
                                f"Type conversion failed for '{col}': {type_e}. Column may contain mixed types or errors.",
                                icon='⚠️',
                            )

                # Store the final combined and processed dataframe
                st.session_state.df = combined_df
                st.session_state.data_loaded = True
                st.session_state.last_files = (
                    selected_files  # Store the actual loaded files
                )

                # Reset view state and filters completely on new data load
                st.session_state.selected_row_index = None
                st.session_state.show_details = False
                st.session_state.current_page = 1
                # Reset filter states to defaults
                st.session_state.filter_sources = []
                st.session_state.filter_step_range = None
                st.session_state.filter_steps_select = []
                st.session_state.filter_reward_range = None
                st.session_state.search_term = ''
                st.session_state.search_prompt = True
                st.session_state.search_completion = True
                # No need to reset filtered_df, it will be recalculated

                status_text.success(
                    f"Loaded {len(st.session_state.df)} samples."
                    + (' Some files had issues.' if load_errors else '')
                )

            except Exception as e:
                st.error(f"Error combining dataframes: {e}")
                st.session_state.data_loaded = False
                st.session_state.df = None
        elif (
            not load_errors
        ):  # No dataframes loaded and no errors reported during load
            st.warning(
                'No data loaded from selected files. Check file content.',
                icon='⚠️',
            )
            st.session_state.data_loaded = False
            st.session_state.df = None
            st.session_state.last_files = (
                selected_files  # Still update last files tried
            )
        else:  # No dataframes loaded, but errors occurred during load
            st.error(
                'Failed to load any valid data. Check file selections and formats.',
                icon='🚨',
            )
            st.session_state.data_loaded = False
            st.session_state.df = None
            st.session_state.last_files = selected_files

        progress_bar.empty()
        status_text.empty()
        # No rerun needed here, script continues and will apply filters


# --- Cached Filtering Function ---
@st.cache_data(ttl=3600)  # Cache for 1 hour, or adjust as needed
def apply_filters_cached(
    df,
    filter_sources,
    filter_step_range,
    filter_steps_select,
    filter_reward_range,
    search_term,
    search_prompt,
    search_completion,
    # Include original min/max for range checks to ensure cache invalidation
    orig_min_step,
    orig_max_step,
    orig_min_reward,
    orig_max_reward,
):
    """Applies filters to the dataframe. Cached by Streamlit."""
    if df is None:
        return pd.DataFrame()  # Return empty df if input is None

    temp_df = df.copy()  # Work on a copy

    # Source filter
    # Convert to tuple for caching if list causes issues (usually not needed)
    if filter_sources:
        temp_df = temp_df[temp_df['source_file'].isin(filter_sources)]

    # Step filter
    if 'step' in temp_df.columns:
        # Range slider filter
        if filter_step_range and isinstance(filter_step_range, tuple):
            min_s, max_s = filter_step_range
            # Apply filter only if the range is different from the original full range
            if (
                orig_min_step is not None
                and orig_max_step is not None
                and (min_s > orig_min_step or max_s < orig_max_step)
            ):
                # Ensure 'step' is numeric for comparison, handling NA
                temp_df = temp_df[
                    pd.to_numeric(temp_df['step'], errors='coerce').between(
                        min_s, max_s, inclusive='both'
                    )
                ]

        # Multiselect step filter (takes precedence if both somehow active)
        elif filter_steps_select:
            # Convert steps_select to numeric type consistent with column if needed
            try:
                select_steps_numeric = pd.to_numeric(
                    filter_steps_select
                ).tolist()
                temp_df = temp_df[
                    pd.to_numeric(temp_df['step'], errors='coerce').isin(
                        select_steps_numeric
                    )
                ]
            except:  # Handle case where conversion fails
                print(
                    'Warning: Could not convert selected steps to numeric for filtering.'
                )
                pass  # Or handle more gracefully

    # Reward filter
    if (
        'accuracy_reward' in temp_df.columns
        and filter_reward_range
        and isinstance(filter_reward_range, tuple)
    ):
        min_r, max_r = filter_reward_range
        # Apply filter only if the range is different from the original full range
        if (
            orig_min_reward is not None
            and orig_max_reward is not None
            and (min_r > orig_min_reward or max_r < orig_max_reward)
        ):
            # Ensure 'accuracy_reward' is numeric for comparison, handling NA
            temp_df = temp_df[
                pd.to_numeric(
                    temp_df['accuracy_reward'], errors='coerce'
                ).between(min_r, max_r, inclusive='both')
            ]

    # Search filter
    if search_term:
        search_conditions = []
        term = search_term  # Use the argument passed to the function
        # Ensure columns exist and search flags are true
        if search_prompt and 'prompt_text' in temp_df.columns:
            # Use regex=False for potentially faster literal string search
            search_conditions.append(
                temp_df['prompt_text']
                .astype(str)
                .str.contains(term, case=False, na=False, regex=False)
            )
        if search_completion and 'completion_text' in temp_df.columns:
            search_conditions.append(
                temp_df['completion_text']
                .astype(str)
                .str.contains(term, case=False, na=False, regex=False)
            )
        if search_conditions:
            # Combine conditions with OR logic
            combined_mask = pd.concat(search_conditions, axis=1).any(axis=1)
            temp_df = temp_df[combined_mask]

    return temp_df


# --- Sidebar Filters ---
filtered_df = pd.DataFrame()  # Initialize as empty
if (
    st.session_state.data_loaded
    and st.session_state.df is not None
    and not st.session_state.df.empty
):
    st.sidebar.subheader('Dataset Statistics')
    st.sidebar.write(f"Total samples loaded: {len(st.session_state.df)}")

    base_df_for_filters = st.session_state.df
    filters_applied_ui = False  # Track if any filter UI is active

    # --- Render Filter Widgets (these update session state via their keys) ---

    # Source file filter
    source_options = []
    if 'source_file' in base_df_for_filters.columns:
        source_options = sorted(
            base_df_for_filters['source_file'].dropna().unique()
        )
        if len(source_options) > 1:
            st.sidebar.multiselect(
                'Filter by source file',
                options=source_options,
                key='filter_sources',  # Let widget update session state directly
            )
            if st.session_state.filter_sources:  # Check if filter is active
                filters_applied_ui = True

    # Step filter - Calculate min/max/unique *once* for widget setup
    min_step, max_step, unique_steps = None, None, []
    orig_min_step, orig_max_step = (
        None,
        None,
    )  # Store original bounds for cache check
    if 'step' in base_df_for_filters.columns:
        valid_steps = (
            pd.to_numeric(base_df_for_filters['step'], errors='coerce')
            .dropna()
            .astype(int)
        )
        if not valid_steps.empty:
            min_step, max_step = int(valid_steps.min()), int(valid_steps.max())
            orig_min_step, orig_max_step = (
                min_step,
                max_step,
            )  # Store original bounds
            unique_steps = sorted(valid_steps.unique())

            st.sidebar.subheader('Filter by Step')
            # Choose slider or multiselect based on number of unique steps
            if len(unique_steps) > 20 and min_step != max_step:
                # Initialize range state if it's None or invalid
                if st.session_state.filter_step_range is None or not (
                    isinstance(st.session_state.filter_step_range, tuple)
                    and len(st.session_state.filter_step_range) == 2
                ):
                    st.session_state.filter_step_range = (min_step, max_step)
                # Ensure the value passed to slider is within the current min/max bounds
                current_range_val = (
                    max(min_step, st.session_state.filter_step_range[0]),
                    min(max_step, st.session_state.filter_step_range[1]),
                )
                st.sidebar.slider(
                    'Step range',
                    min_value=min_step,
                    max_value=max_step,
                    value=current_range_val,
                    key='filter_step_range',  # Updates session state
                )
                # Check if the slider is applying a filter
                if st.session_state.filter_step_range != (min_step, max_step):
                    filters_applied_ui = True
                    st.session_state.filter_steps_select = (
                        []
                    )  # Clear multiselect if slider used

            elif len(unique_steps) > 1:
                # Ensure default is a subset of available unique steps
                valid_default_steps = [
                    s
                    for s in st.session_state.filter_steps_select
                    if s in unique_steps
                ]
                st.sidebar.multiselect(
                    'Select steps',
                    unique_steps,
                    default=valid_default_steps,
                    key='filter_steps_select',  # Updates session state
                )
                if st.session_state.filter_steps_select:
                    filters_applied_ui = True
                    st.session_state.filter_step_range = (
                        None  # Clear slider if multiselect used
                    )

    # Accuracy reward filter - Calculate min/max *once*
    min_reward, max_reward = None, None
    orig_min_reward, orig_max_reward = (
        None,
        None,
    )  # Store original bounds for cache check
    if 'accuracy_reward' in base_df_for_filters.columns:
        valid_rewards = pd.to_numeric(
            base_df_for_filters['accuracy_reward'], errors='coerce'
        ).dropna()
        if not valid_rewards.empty:
            min_reward, max_reward = float(valid_rewards.min()), float(
                valid_rewards.max()
            )
            orig_min_reward, orig_max_reward = (
                min_reward,
                max_reward,
            )  # Store original bounds

            st.sidebar.subheader('Filter by Accuracy Reward')
            if min_reward != max_reward:
                # Initialize range state if it's None or invalid
                if st.session_state.filter_reward_range is None or not (
                    isinstance(st.session_state.filter_reward_range, tuple)
                    and len(st.session_state.filter_reward_range) == 2
                ):
                    st.session_state.filter_reward_range = (
                        min_reward,
                        max_reward,
                    )

                # Ensure the value passed to slider is within the current min/max bounds
                current_reward_val = (
                    max(min_reward, st.session_state.filter_reward_range[0]),
                    min(max_reward, st.session_state.filter_reward_range[1]),
                )
                st.sidebar.slider(
                    'Accuracy reward range',
                    min_value=min_reward,
                    max_value=max_reward,
                    value=current_reward_val,
                    key='filter_reward_range',  # Updates session state
                )
                # Check if the slider is applying a filter
                if st.session_state.filter_reward_range != (
                    min_reward,
                    max_reward,
                ):
                    filters_applied_ui = True

    # Search filter
    st.sidebar.subheader('Search Text')
    st.sidebar.text_input('Search prompts/completions', key='search_term')
    st.sidebar.checkbox('Search in prompts', key='search_prompt')
    st.sidebar.checkbox('Search in completions', key='search_completion')
    if st.session_state.search_term:
        filters_applied_ui = True

    # --- Apply Filters using Cached Function ---
    # Pass filter states from session_state to the cached function
    # Use tuples for lists passed to cache function if needed, but usually works
    filtered_df = apply_filters_cached(
        st.session_state.df,
        tuple(st.session_state.filter_sources),  # Pass as tuple
        st.session_state.filter_step_range,
        tuple(st.session_state.filter_steps_select),  # Pass as tuple
        st.session_state.filter_reward_range,
        st.session_state.search_term,
        st.session_state.search_prompt,
        st.session_state.search_completion,
        # Pass original bounds for cache invalidation checks
        orig_min_step,
        orig_max_step,
        orig_min_reward,
        orig_max_reward,
    )

    # --- Post-filtering State Updates ---
    # Check if the currently selected row is still valid in the new filtered_df
    if (
        st.session_state.selected_row_index is not None
        and st.session_state.selected_row_index not in filtered_df.index
    ):
        # If the selected row is gone, hide details and reset selection
        st.session_state.show_details = False
        st.session_state.selected_row_index = None
        # Don't rerun here, let the rest of the script execute with the updated state

    # Check if the current page number is still valid
    rows_per_page = st.session_state.rows_per_page
    total_rows = len(filtered_df)
    max_pages = max(1, (total_rows + rows_per_page - 1) // rows_per_page)
    if st.session_state.current_page > max_pages:
        st.session_state.current_page = (
            max_pages  # Adjust page number if needed
        )
    if st.session_state.current_page < 1:
        st.session_state.current_page = 1

    # Display filter status and reset button
    st.sidebar.write(f"Filtered samples: {len(filtered_df)}")
    if filters_applied_ui:
        if st.sidebar.button('Reset All Filters'):
            # Reset all filter states
            st.session_state.filter_sources = []
            st.session_state.filter_step_range = None
            st.session_state.filter_steps_select = []
            st.session_state.filter_reward_range = None
            st.session_state.search_term = ''
            st.session_state.search_prompt = True
            st.session_state.search_completion = True
            # Reset view state as well
            st.session_state.current_page = 1
            st.session_state.selected_row_index = None
            st.session_state.show_details = False
            # Clear the cache for the filter function explicitly if desired,
            # though changing inputs should invalidate it anyway.
            # apply_filters_cached.clear() # Optional: force clear cache
            st.rerun()  # Rerun to apply the reset state

# --- Main Content ---
st.title('LLM Sample Analyzer')

# Use the filtered dataframe obtained from the cached function
# Check if data is loaded and df exists before proceeding
if st.session_state.data_loaded and isinstance(filtered_df, pd.DataFrame):

    tab1, tab2 = st.tabs(['Samples Overview', 'Analytics'])

    with tab1:
        # --- Layout Definition ---
        # Determine layout based on whether details should be shown
        # Check if selected_row_index is valid *within the current filtered_df*
        is_detail_view_valid = (
            st.session_state.show_details
            and st.session_state.selected_row_index is not None
            and st.session_state.selected_row_index in filtered_df.index
        )

        if is_detail_view_valid:
            col_table, col_details = st.columns(
                [1, 1]
            )  # Adjust ratio if needed e.g., [3, 2]
        else:
            # If index became invalid after filtering, don't show details pane
            col_table = st
            col_details = None
            if (
                st.session_state.show_details
                and st.session_state.selected_row_index is not None
            ):
                # Display a message if details were expected but index is now invalid
                st.warning(
                    f"Previously selected sample index {st.session_state.selected_row_index} is no longer available with current filters.",
                    icon='⚠️',
                )
                # Automatically hide details panel in this case
                st.session_state.show_details = False
                st.session_state.selected_row_index = None

        # --- Table View (col_table) using Loop and Columns ---
        col_table.subheader('Samples')
        if not filtered_df.empty:
            # Pagination Controls
            col_table.select_slider(
                'Rows per page', [5, 10, 20, 50, 100], key='rows_per_page'
            )
            rows_per_page = st.session_state.rows_per_page
            total_rows = len(filtered_df)
            max_pages = max(
                1, (total_rows + rows_per_page - 1) // rows_per_page
            )

            # Ensure current page is valid (already done after filtering, but good practice)
            st.session_state.current_page = max(
                1, min(st.session_state.current_page, max_pages)
            )

            nav_cols = col_table.columns([1, 2, 1])  # Prev, Page Info, Next

            # Previous Button
            if nav_cols[0].button(
                '◀ Prev',
                disabled=(st.session_state.current_page <= 1),
                use_container_width=True,
                key='prev_page_button',
            ):
                st.session_state.current_page -= 1
                st.rerun()  # Rerun to display the new page (will be fast due to cache)

            # Page Number Input and Display
            # Using number input can cause multiple reruns while typing.
            # Displaying text is safer. Consider a dedicated page input later if needed.
            nav_cols[1].markdown(
                f"<div style='text-align: center;'>Page **{st.session_state.current_page}** of **{max_pages}**</div>",
                unsafe_allow_html=True,
            )
            # Optional: Add number input if precise page jumping is critical
            # page_num_input = nav_cols[1].number_input(...)

            # Next Button
            if nav_cols[2].button(
                'Next ▶',
                disabled=(st.session_state.current_page >= max_pages),
                use_container_width=True,
                key='next_page_button',
            ):
                st.session_state.current_page += 1
                st.rerun()  # Rerun to display the new page (will be fast due to cache)

            # Calculate indices for the current page
            start_idx_pos = (st.session_state.current_page - 1) * rows_per_page
            end_idx_pos = min(start_idx_pos + rows_per_page, total_rows)
            col_table.write(
                f"Showing samples {start_idx_pos+1} - {end_idx_pos} of {total_rows}"
            )

            # Get the slice of data for the current page using positional slicing (.iloc)
            page_df = filtered_df.iloc[start_idx_pos:end_idx_pos]

            # Display rows using columns for better structure
            for i, (idx, row) in enumerate(
                page_df.iterrows()
            ):  # idx is the original index from df
                with col_table.container():
                    # Use markdown for a subtle separator
                    st.markdown(
                        '<hr style="margin-top: 0.5rem; margin-bottom: 0.5rem; border-top: 1px solid var(--content-border-color);">',
                        unsafe_allow_html=True,
                    )
                    row_cols = st.columns(
                        [4.5, 0.5]
                    )  # Ratio for content vs button

                    # --- Left Column (Info & Previews) ---
                    info_parts = [f"Index: {idx}"]  # Always show original index
                    if 'step' in row and pd.notna(row['step']):
                        info_parts.append(f"Step: {int(row['step'])}")
                    if 'accuracy_reward' in row and pd.notna(
                        row['accuracy_reward']
                    ):
                        info_parts.append(
                            f"Reward: {row['accuracy_reward']:.3f}"
                        )
                    if 'prompt_length' in row and pd.notna(
                        row['prompt_length']
                    ):
                        info_parts.append(
                            f"Prompt Len: {int(row['prompt_length'])}"
                        )
                    if 'completion_length' in row and pd.notna(
                        row['completion_length']
                    ):
                        info_parts.append(
                            f"Comp Len: {int(row['completion_length'])}"
                        )
                    if 'source_file' in row and pd.notna(row['source_file']):
                        info_parts.append(f"Source: {row['source_file']}")

                    row_cols[0].markdown(
                        f"<div class='sample-info'>{' | '.join(info_parts)}</div>",
                        unsafe_allow_html=True,
                    )

                    # --- Prompt Preview ---
                    prompt_text = row.get('prompt_text', None)
                    if pd.notna(prompt_text) and prompt_text:
                        # Use helper, escape handled inside truncate/highlight
                        preview = truncate_middle(prompt_text, max_length=250)
                        preview_highlighted = highlight_text(
                            preview,  # Pass already truncated text to highlight
                            (
                                st.session_state.search_term
                                if st.session_state.search_prompt
                                else ''
                            ),
                        )
                        row_cols[0].markdown(
                            f"<div class='preview-label'>Prompt:</div><div class='sample-prompt-preview'>{preview_highlighted}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        row_cols[0].markdown(
                            "<div class='preview-label'>Prompt:</div><div class='sample-prompt-preview'><i>N/A</i></div>",
                            unsafe_allow_html=True,
                        )

                    # --- Completion Preview ---
                    completion_text = row.get('completion_text', None)
                    if pd.notna(completion_text) and completion_text:
                        preview = truncate_middle(
                            completion_text, max_length=1000
                        )
                        preview_highlighted = highlight_text(
                            preview,  # Pass already truncated text to highlight
                            (
                                st.session_state.search_term
                                if st.session_state.search_completion
                                else ''
                            ),
                        )
                        row_cols[0].markdown(
                            f"<div class='preview-label'>Completion:</div><div class='sample-completion-preview'>{preview_highlighted}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        row_cols[0].markdown(
                            "<div class='preview-label'>Completion:</div><div class='sample-completion-preview'><i>N/A</i></div>",
                            unsafe_allow_html=True,
                        )

                    # --- Right Column (Button) ---
                    button_key = f"view_details_{idx}"  # Unique key per button
                    if row_cols[1].button(
                        'View Details',
                        key=button_key,
                        help=f"View details for sample index {idx}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_row_index = idx
                        st.session_state.show_details = True
                        # No need to set other flags
                        st.rerun()  # Rerun to show details panel (will be fast due to cache)

            # Separator after the list
            col_table.markdown(
                '<hr style="margin-top: 0.5rem; margin-bottom: 0.5rem; border-top: 1px solid var(--content-border-color);">',
                unsafe_allow_html=True,
            )

            # Export Button
            try:
                # Use the filtered_df for export
                csv_data = filtered_df.to_csv(index=False).encode('utf-8')
                col_table.download_button(
                    label='Export Filtered Data to CSV',
                    data=csv_data,
                    file_name=f"filtered_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime='text/csv',
                    key='export_csv',
                )
            except Exception as e:
                col_table.warning(
                    f"Could not prepare CSV for download: {e}", icon='⚠️'
                )

        else:
            col_table.warning('No samples match the current filters.', icon='⚠️')

        # --- Details Panel (col_details) ---
        # Check again if the detail view is valid before rendering
        if is_detail_view_valid and col_details is not None:
            try:
                # Get the selected sample data directly using the index
                sample = filtered_df.loc[st.session_state.selected_row_index]

                # Find position within the *current* filtered dataframe for nav
                # Get the index list of the *current* filtered dataframe
                filtered_indices = filtered_df.index.tolist()
                try:
                    # Find the positional index (0, 1, 2...) of the selected row's original index
                    sample_pos_idx = filtered_indices.index(
                        st.session_state.selected_row_index
                    )
                except ValueError:
                    # This should ideally not happen due to the is_detail_view_valid check
                    col_details.error(
                        'Selected index inconsistency. Please refresh or select again.',
                        icon='🚨',
                    )
                    st.session_state.show_details = False
                    st.session_state.selected_row_index = None
                    st.rerun()  # Rerun to clear the invalid state

                # Header with close button
                detail_header_cols = col_details.columns([5, 1])
                detail_header_cols[0].header(
                    f"Sample Details (Index: {st.session_state.selected_row_index})"
                )
                if detail_header_cols[1].button(
                    '✖', help='Close details', key='close_details'
                ):
                    st.session_state.show_details = False
                    st.session_state.selected_row_index = None
                    st.rerun()  # Rerun to hide the panel

                # Display Step Info
                if 'step' in sample and pd.notna(sample['step']):
                    col_details.markdown(
                        f"<div class='section-header'>Step: {int(sample['step'])}</div>",
                        unsafe_allow_html=True,
                    )

                # Ground Truth and Accuracy Reward
                gt_acc_cols = col_details.columns(2)
                gt_acc_cols[0].markdown(
                    '<strong>Ground Truth</strong>', unsafe_allow_html=True
                )
                # Use .get with default, ensure string conversion and escape
                gt_content = html.escape(str(sample.get('ground_truth', 'N/A')))
                gt_acc_cols[0].markdown(
                    f"<div class='content-container content-text'>{gt_content}</div>",
                    unsafe_allow_html=True,
                )

                gt_acc_cols[1].markdown(
                    '<strong>Accuracy Reward</strong>', unsafe_allow_html=True
                )
                acc_val = sample.get('accuracy_reward', None)
                if pd.notna(acc_val):
                    # Ensure acc_val is float for comparison
                    try:
                        acc_val_float = float(acc_val)
                        status_class = (
                            'status-good'
                            if acc_val_float > 0.75
                            else (
                                'status-bad'
                                if acc_val_float < 0.25
                                else 'status-neutral'
                            )
                        )
                        gt_acc_cols[1].markdown(
                            f"<div class='status-base {status_class}'>{acc_val_float:.3f}</div>",
                            unsafe_allow_html=True,
                        )
                    except (ValueError, TypeError):
                        gt_acc_cols[1].markdown(
                            f"<div class='content-container content-text'>{html.escape(str(acc_val))}</div>",  # Show raw value if not numeric
                            unsafe_allow_html=True,
                        )
                else:
                    gt_acc_cols[1].markdown(
                        "<div class='content-container content-text'>N/A</div>",
                        unsafe_allow_html=True,
                    )

                # Metadata Expander
                with col_details.expander('Metadata', expanded=False):
                    metadata_html = "<div class='content-container'>"
                    exclude_cols = [  # Define columns to exclude from metadata display
                        'prompt_text',
                        'completion_text',
                        'ground_truth',
                        'accuracy_reward',
                        'step',  # Already displayed prominently
                        # Add any other internal/preview columns if they exist
                    ]
                    items_found = False
                    # Iterate through Series items (column name, value)
                    for key, value in sample.items():
                        # Check if key should be excluded and if value is not null/NA
                        if key not in exclude_cols and pd.notna(value):
                            # Escape both key and value for security
                            metadata_html += f"<div class='metadata-item'><span class='metadata-label'>{html.escape(str(key))}:</span> {html.escape(str(value))}</div>"
                            items_found = True
                    if not items_found:
                        metadata_html += "<div class='metadata-item'>No additional metadata found.</div>"
                    metadata_html += '</div>'
                    st.markdown(metadata_html, unsafe_allow_html=True)

                # Prompt Text
                col_details.markdown(
                    "<div class='section-header'>Prompt</div>",
                    unsafe_allow_html=True,
                )
                prompt_text_detail = sample.get('prompt_text', 'N/A')
                # Highlight the full text in the details view
                highlighted_prompt = highlight_text(
                    prompt_text_detail,
                    (
                        st.session_state.search_term
                        if st.session_state.search_prompt
                        else ''
                    ),
                )
                col_details.markdown(
                    f"<div class='content-container'><div class='content-text'>{highlighted_prompt}</div></div>",
                    unsafe_allow_html=True,
                )

                # Completion Text
                col_details.markdown(
                    "<div class='section-header'>Completion</div>",
                    unsafe_allow_html=True,
                )
                completion_text_detail = sample.get('completion_text', 'N/A')
                highlighted_completion = highlight_text(
                    completion_text_detail,
                    (
                        st.session_state.search_term
                        if st.session_state.search_completion
                        else ''
                    ),
                )
                col_details.markdown(
                    f"<div class='content-container'><div class='content-text'>{highlighted_completion}</div></div>",
                    unsafe_allow_html=True,
                )

                # Detail Navigation Buttons
                nav_cols_detail = col_details.columns(2)
                # Check position within the filtered list
                prev_disabled = sample_pos_idx <= 0
                if nav_cols_detail[0].button(
                    '⬅ Previous Sample',
                    disabled=prev_disabled,
                    use_container_width=True,
                    key='prev_sample',
                ):
                    if not prev_disabled:
                        # Get the original index of the previous item in the filtered list
                        st.session_state.selected_row_index = filtered_indices[
                            sample_pos_idx - 1
                        ]
                        # st.session_state.show_details remains True
                        st.rerun()  # Rerun to show previous sample (fast due to cache)

                next_disabled = sample_pos_idx >= len(filtered_indices) - 1
                if nav_cols_detail[1].button(
                    'Next Sample ➡',
                    disabled=next_disabled,
                    use_container_width=True,
                    key='next_sample',
                ):
                    if not next_disabled:
                        # Get the original index of the next item in the filtered list
                        st.session_state.selected_row_index = filtered_indices[
                            sample_pos_idx + 1
                        ]
                        # st.session_state.show_details remains True
                        st.rerun()  # Rerun to show next sample (fast due to cache)

            except KeyError:
                # This might happen if the index was valid moments ago but something changed unexpectedly
                col_details.error(
                    f"Error: Sample index {st.session_state.selected_row_index} could not be accessed.",
                    icon='🚨',
                )
                st.session_state.show_details = False
                st.session_state.selected_row_index = None
                st.rerun()  # Attempt to recover state
            except Exception as e:
                col_details.error(
                    f"An unexpected error occurred displaying details: {e}",
                    icon='🚨',
                )
                # Optionally reset state on unexpected error
                # st.session_state.show_details = False
                # st.session_state.selected_row_index = None

    with tab2:
        st.header('Analytics')
        if not filtered_df.empty:
            st.write('Basic plots based on the **filtered** data.')

            plot_cols = st.columns(2)

            # Plot 1: Accuracy Reward Distribution (if column exists)
            if (
                'accuracy_reward' in filtered_df.columns
                and filtered_df['accuracy_reward'].notna().any()
            ):
                try:
                    fig_reward = px.histogram(
                        filtered_df,
                        x='accuracy_reward',
                        title='Accuracy Reward Distribution',
                        nbins=30,
                    )
                    fig_reward.update_layout(bargap=0.1)
                    plot_cols[0].plotly_chart(
                        fig_reward, use_container_width=True
                    )
                except Exception as e:
                    plot_cols[0].warning(f"Could not plot accuracy reward: {e}")
            else:
                plot_cols[0].info(
                    "No 'accuracy_reward' data available for plotting."
                )

            # Plot 2: Prompt Length Distribution (if column exists)
            if (
                'prompt_length' in filtered_df.columns
                and filtered_df['prompt_length'].notna().any()
            ):
                try:
                    fig_prompt_len = px.histogram(
                        filtered_df,
                        x='prompt_length',
                        title='Prompt Length Distribution',
                        nbins=30,
                    )
                    fig_prompt_len.update_layout(bargap=0.1)
                    plot_cols[1].plotly_chart(
                        fig_prompt_len, use_container_width=True
                    )
                except Exception as e:
                    plot_cols[1].warning(f"Could not plot prompt length: {e}")
            else:
                plot_cols[1].info(
                    "No 'prompt_length' data available for plotting."
                )

            # Plot 3: Completion Length Distribution (if column exists)
            if (
                'completion_length' in filtered_df.columns
                and filtered_df['completion_length'].notna().any()
            ):
                try:
                    fig_comp_len = px.histogram(
                        filtered_df,
                        x='completion_length',
                        title='Completion Length Distribution',
                        nbins=30,
                    )
                    fig_comp_len.update_layout(bargap=0.1)
                    plot_cols[0].plotly_chart(
                        fig_comp_len, use_container_width=True
                    )
                except Exception as e:
                    plot_cols[0].warning(
                        f"Could not plot completion length: {e}"
                    )
            else:
                plot_cols[0].info(
                    "No 'completion_length' data available for plotting."
                )

            # Plot 4: Reward vs Step (if columns exist)
            if (
                'step' in filtered_df.columns
                and 'accuracy_reward' in filtered_df.columns
                and filtered_df['step'].notna().any()
                and filtered_df['accuracy_reward'].notna().any()
            ):
                try:
                    # Aggregate data for scatter plot if too many points
                    plot_df_agg = filtered_df.dropna(
                        subset=['step', 'accuracy_reward']
                    )
                    if len(plot_df_agg) > 5000:  # Limit points for performance
                        plot_df_agg = plot_df_agg.sample(n=5000, random_state=1)

                    fig_reward_step = px.scatter(
                        plot_df_agg,
                        x='step',
                        y='accuracy_reward',
                        title='Accuracy Reward vs. Step',
                        opacity=0.6,
                        trendline='lowess',  # Add a smoothed trendline
                        trendline_options=dict(frac=0.1),
                    )  # Adjust smoothing
                    plot_cols[1].plotly_chart(
                        fig_reward_step, use_container_width=True
                    )
                except Exception as e:
                    plot_cols[1].warning(f"Could not plot reward vs step: {e}")
            else:
                plot_cols[1].info(
                    "Need 'step' and 'accuracy_reward' data for scatter plot."
                )

        else:
            st.info(
                'No data available to display analytics. Apply different filters or load data.'
            )


# --- Initial State / No Data Loaded ---
elif not selected_files:
    st.info('Select data source and files from the sidebar to begin analysis.')
    # How to Use section remains the same...
    st.markdown(
        """
     ### How to Use This Tool

     #### Option 1: Load from Directory
     1. Select "Directory" from the sidebar.
     2. Enter the path to a directory containing JSONL or Parquet files.
     3. Select which files you want to analyze from the list that appears.

     #### Option 2: Specify Exact Files
     1. Select "Specific Files" from the sidebar.
     2. Enter the full paths to your JSONL/Parquet files, one path per line, in the text area.
     3. Confirm the files you want to analyze from the list that appears.

     #### Expected File Format
     The application works best with files containing records (one per line in JSONL) like:

     ```json
     {
         "step": 1000,
         "prompt_text": "What is the capital of France?",
         "completion_text": "The capital of France is Paris.",
         "accuracy_reward": 1.0,
         "prompt_length": 31,
         "completion_length": 31,
         "ground_truth": "Paris",
         "source_file": "optional_filename.jsonl" /* Added by the app if loading multiple files */
     }
     ```
     *Note: Not all fields are required, but `prompt_text` and `completion_text` are recommended for viewing details. Analytics plots require relevant columns like `step`, `accuracy_reward`, `prompt_length`, `completion_length`.*

     #### Running from Command Line
     You can pre-fill the directory path when launching:

     ```bash
     streamlit run analyze_app.py -- --data_dir=/path/to/your/data
     ```
     """
    )
else:  # Handles the case where data loading failed or resulted in an empty df
    st.error(
        'Data could not be loaded or no valid data found. Please check file selections, paths, and formats in the sidebar.',
        icon='🚨',
    )
