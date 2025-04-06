# flake8: noqa


import glob
import html
import json
import os
import re
import sys
from datetime import datetime

import pandas as pd
import plotly.express as px
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
        max-height: 4.2em; /* Limit height to roughly 3 lines */
        overflow: hidden; /* Hide overflow */
        display: -webkit-box; /* Enable flexbox */
        -webkit-line-clamp: 3; /* Limit to 3 lines */
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
                        st.warning(
                            f"Skipping invalid JSON line {i + 1} in {os.path.basename(file_path)}: {line[:100]}... Error: {json_err}",
                            icon='⚠️',
                        )
                        continue
            if records:
                return pd.DataFrame(records)
            return pd.DataFrame()
        elif file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        else:
            st.error(
                f"Unsupported file format: {file_path}. Please use .jsonl or .parquet."
            )
            return None
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return None
    except Exception as e:
        st.error(f"Error loading file {file_path}: {e}")
        return None


def highlight_text(text, search_term):
    """Highlights search term in text using HTML span, escaping the text first."""
    if not search_term or not isinstance(text, str):
        return html.escape(str(text))  # Escape non-strings or if no search term
    try:
        escaped_text = html.escape(text)
        # Use word boundaries to avoid highlighting parts of words unless the search term itself has spaces/punctuation
        # If search term has special regex chars, escape them
        escaped_search_term = re.escape(search_term)
        # Simple highlighting without word boundaries if needed:
        # highlighted = re.sub(
        #     f"({escaped_search_term})",
        #     r"<span class='search-highlight'>\1</span>",
        #     escaped_text,
        #     flags=re.IGNORECASE,
        # )
        # Highlighting with word boundaries (more precise for whole words):
        highlighted = re.sub(
            r'\b(' + escaped_search_term + r')\b',
            r"<span class='search-highlight'>\1</span>",
            escaped_text,
            flags=re.IGNORECASE,
        )
        # Fallback if no word boundary match (e.g., searching for punctuation)
        if highlighted == escaped_text:
            highlighted = re.sub(
                f"({escaped_search_term})",
                r"<span class='search-highlight'>\1</span>",
                escaped_text,
                flags=re.IGNORECASE,
            )

        return highlighted
    except Exception as e:
        st.warning(f"Error highlighting text: {e}", icon='⚠️')
        return html.escape(text)  # Return escaped text on error


# --- NEW HELPER FUNCTION ---
def truncate_middle(text, max_length=200, placeholder=' ... '):
    """Truncates text showing the start and end, joined by a placeholder."""
    if not isinstance(text, str) or len(text) <= max_length:
        return html.escape(str(text))  # Escape even if not truncated

    placeholder_len = len(placeholder)
    if max_length <= placeholder_len:
        # Not enough space for placeholder, just truncate from end
        return html.escape(text[:max_length])

    chars_to_keep = max_length - placeholder_len
    # Prioritize showing a bit more of the beginning
    front_chars = (chars_to_keep + 1) // 2
    back_chars = chars_to_keep - front_chars

    # Ensure back_chars is positive
    if back_chars <= 0:
        front_chars = max_length - placeholder_len
        return html.escape(text[:front_chars]) + placeholder

    start_part = html.escape(text[:front_chars].strip())
    end_part = html.escape(text[-back_chars:].strip())

    # Add placeholder only if there's a gap
    if len(text) > front_chars + back_chars:
        return f"{start_part}{placeholder}{end_part}"
    else:  # Should not happen with initial check, but as safety
        return html.escape(text)


# --- Session State Initialization ---
def init_session_state():
    defaults = {
        'data_loaded': False,
        'selected_row_index': None,
        'show_details': False,
        'filtered_df': None,  # Will store the filtered dataframe
        'df': None,  # Will store the original loaded dataframe
        'current_page': 1,
        'last_files': [],
        'data_dir': os.getcwd(),
        'specific_files_input': '',
        'selected_files_cache': [],
        'selected_specific_files_cache': [],
        'filter_sources': [],
        'filter_step_range': None,
        'filter_steps_select': [],
        'filter_reward_range': None,
        'search_term': '',
        'search_prompt': True,
        'search_completion': True,
        'rows_per_page': 10,
        'view_button_just_clicked': False,
        'filters_changed_flag': False,  # Flag to explicitly track filter changes
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
# (No changes needed here)
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
    if data_dir_input != st.session_state.data_dir and os.path.isdir(
        data_dir_input
    ):
        st.session_state.data_dir = data_dir_input
        st.session_state.selected_files_cache = []
        st.rerun()
    data_dir = st.session_state.data_dir

    if data_dir and os.path.isdir(data_dir):
        try:
            all_files = sorted(
                glob.glob(os.path.join(data_dir, '*.jsonl'))
                + glob.glob(os.path.join(data_dir, '*.parquet'))
            )
            if all_files:
                valid_defaults = [
                    f
                    for f in st.session_state.selected_files_cache
                    if f in all_files
                ] or all_files[: min(5, len(all_files))]
                selected_files = st.sidebar.multiselect(
                    'Select files',
                    all_files,
                    default=valid_defaults,
                    key='file_multiselect_dir',
                )
                if selected_files != st.session_state.selected_files_cache:
                    st.session_state.selected_files_cache = selected_files
            else:
                st.sidebar.info(
                    f"No JSONL or Parquet files found in '{os.path.basename(data_dir)}'."
                )
        except Exception as e:
            st.sidebar.error(f"Error accessing directory: {e}")
    elif data_dir:
        st.sidebar.error(f"Directory not found: '{data_dir}'")

else:  # Specific Files
    file_paths_input = st.sidebar.text_area(
        'Enter file paths (one per line)',
        value=st.session_state.specific_files_input,
        key='specific_files_text_area',
    )
    if file_paths_input != st.session_state.specific_files_input:
        st.session_state.specific_files_input = file_paths_input
        st.session_state.selected_specific_files_cache = []
        st.rerun()

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
            valid_defaults = [
                f
                for f in st.session_state.selected_specific_files_cache
                if f in valid_paths
            ] or valid_paths
            selected_files = st.sidebar.multiselect(
                'Confirm files',
                valid_paths,
                default=valid_defaults,
                key='file_multiselect_specific',
            )
            if selected_files != st.session_state.selected_specific_files_cache:
                st.session_state.selected_specific_files_cache = selected_files
        elif not invalid_paths:
            st.sidebar.info('Enter valid file paths above.')

# --- Data Loading ---
# (Logic slightly adjusted to trigger initial filtering)
data_load_trigger = False
if selected_files and (
    not st.session_state.data_loaded
    or st.session_state.last_files != selected_files
):
    with st.spinner('Loading data...'):
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        loaded_dataframes = []
        total_files = len(selected_files)

        for i, file_path in enumerate(selected_files):
            base_name = os.path.basename(file_path)
            status_text.text(f"Loading {base_name} ({i + 1}/{total_files})...")
            df = load_data_from_path(file_path)
            if df is not None and not df.empty:
                df['source_file'] = base_name
                expected_cols = [
                    'step',
                    'prompt_text',
                    'completion_text',
                    'accuracy_reward',
                    'prompt_length',
                    'completion_length',
                    'ground_truth',
                ]
                for col in expected_cols:
                    if col not in df.columns:
                        df[col] = pd.NA
                if (
                    'prompt_length' not in df.columns
                    and 'prompt_text' in df.columns
                ):
                    df['prompt_length'] = (
                        df['prompt_text'].astype(str).apply(len)
                    )
                if (
                    'completion_length' not in df.columns
                    and 'completion_text' in df.columns
                ):
                    df['completion_length'] = (
                        df['completion_text'].astype(str).apply(len)
                    )
                loaded_dataframes.append(df)
            progress_bar.progress((i + 1) / total_files)

        status_text.text('Combining data...')
        if loaded_dataframes:
            try:
                st.session_state.df = pd.concat(
                    loaded_dataframes, ignore_index=True
                )
                col_types = {
                    'step': 'Int64',
                    'accuracy_reward': 'float64',
                    'prompt_length': 'Int64',
                    'completion_length': 'Int64',
                }
                for col, dtype in col_types.items():
                    if col in st.session_state.df.columns:
                        try:
                            st.session_state.df[col] = pd.to_numeric(
                                st.session_state.df[col], errors='coerce'
                            )
                            if dtype == 'Int64':
                                st.session_state.df[col] = st.session_state.df[
                                    col
                                ].astype(dtype)
                        except Exception as type_e:
                            st.warning(
                                f"Type conversion failed for '{col}': {type_e}",
                                icon='⚠️',
                            )

                st.session_state.data_loaded = True
                st.session_state.last_files = selected_files
                st.session_state.selected_row_index = None
                st.session_state.show_details = False
                st.session_state.current_page = 1
                st.session_state.filter_step_range = None
                st.session_state.filter_reward_range = None
                st.session_state.filters_changed_flag = (
                    True  # Force filter application on new data
                )
                data_load_trigger = True  # Signal that data was just loaded
                status_text.success(
                    f"Loaded {len(st.session_state.df)} samples."
                )

            except Exception as e:
                st.error(f"Error combining dataframes: {e}")
                st.session_state.data_loaded = False
        else:
            st.warning('No data loaded from selected files.', icon='⚠️')
            st.session_state.data_loaded = False
            st.session_state.df = None
            st.session_state.filtered_df = None

        progress_bar.empty()
        status_text.empty()

# --- Sidebar Filters (Refactored for Performance) ---
if (
    st.session_state.data_loaded and st.session_state.df is not None
):  # noqa: C901
    st.sidebar.subheader('Dataset Statistics')
    st.sidebar.write(f"Total samples loaded: {len(st.session_state.df)}")

    # --- Store state BEFORE rendering filter widgets ---
    original_filter_state = {
        'sources': list(st.session_state.filter_sources),
        'step_range': st.session_state.filter_step_range,
        'steps_select': list(st.session_state.filter_steps_select),
        'reward_range': st.session_state.filter_reward_range,
        'search': st.session_state.search_term,
        'search_p': st.session_state.search_prompt,
        'search_c': st.session_state.search_completion,
    }

    # --- Render Filter Widgets (these update session state via their keys) ---
    base_df_for_filters = (
        st.session_state.df
    )  # Use original df to determine filter options
    filters_applied_ui = False  # Track if any filter UI is active

    # Source file filter
    if (
        'source_file' in base_df_for_filters.columns
        and base_df_for_filters['source_file'].nunique() > 1
    ):
        st.sidebar.multiselect(
            'Filter by source file',
            sorted(base_df_for_filters['source_file'].unique()),
            key='filter_sources',
        )
        if st.session_state.filter_sources:
            filters_applied_ui = True

    # Step filter
    if (
        'step' in base_df_for_filters.columns
        and not base_df_for_filters['step'].isnull().all()
    ):
        valid_steps = base_df_for_filters['step'].dropna().astype(int)
        if not valid_steps.empty:
            min_step, max_step = int(valid_steps.min()), int(valid_steps.max())
            unique_steps = sorted(valid_steps.unique())
            st.sidebar.subheader('Filter by Step')

            if len(unique_steps) > 20 and min_step != max_step:
                current_range = st.session_state.filter_step_range
                if current_range is None or not (
                    min_step <= current_range[0] <= current_range[1] <= max_step
                ):
                    st.session_state.filter_step_range = (min_step, max_step)
                st.sidebar.slider(
                    'Step range',
                    min_value=min_step,
                    max_value=max_step,
                    key='filter_step_range',
                )
                if st.session_state.filter_step_range != (min_step, max_step):
                    filters_applied_ui = True
            elif len(unique_steps) > 1:
                st.sidebar.multiselect(
                    'Select steps', unique_steps, key='filter_steps_select'
                )
                if st.session_state.filter_steps_select:
                    filters_applied_ui = True

    # Accuracy reward filter
    if (
        'accuracy_reward' in base_df_for_filters.columns
        and not base_df_for_filters['accuracy_reward'].isnull().all()
    ):
        valid_rewards = base_df_for_filters['accuracy_reward'].dropna()
        if not valid_rewards.empty:
            min_reward, max_reward = float(valid_rewards.min()), float(
                valid_rewards.max()
            )
            st.sidebar.subheader('Filter by Accuracy Reward')
            if min_reward != max_reward:
                current_reward_range = st.session_state.filter_reward_range
                epsilon = 1e-9
                if current_reward_range is None or not (
                    min_reward - epsilon
                    <= current_reward_range[0]
                    <= current_reward_range[1]
                    <= max_reward + epsilon
                ):
                    st.session_state.filter_reward_range = (
                        min_reward,
                        max_reward,
                    )
                st.sidebar.slider(
                    'Accuracy reward range',
                    min_value=min_reward,
                    max_value=max_reward,
                    key='filter_reward_range',
                )
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

    # --- Check if Filters Changed ---
    current_filter_state = {
        'sources': list(st.session_state.filter_sources),
        'step_range': st.session_state.filter_step_range,
        'steps_select': list(st.session_state.filter_steps_select),
        'reward_range': st.session_state.filter_reward_range,
        'search': st.session_state.search_term,
        'search_p': st.session_state.search_prompt,
        'search_c': st.session_state.search_completion,
    }

    # Determine if filters actually changed compared to the start of the script run
    filters_changed = current_filter_state != original_filter_state
    if filters_changed:
        st.session_state.filters_changed_flag = True

    # --- Apply Filters Conditionally ---
    # Apply filters only if the flag is set (due to UI change or new data load)
    # OR if filtered_df hasn't been created yet.
    if (
        st.session_state.filters_changed_flag
        or 'filtered_df' not in st.session_state
        or st.session_state.filtered_df is None
    ):
        with st.spinner('Applying filters...'):
            temp_filtered_df = (
                st.session_state.df.copy()
            )  # Start fresh from original data

            # Apply source filter
            if st.session_state.filter_sources:
                temp_filtered_df = temp_filtered_df[
                    temp_filtered_df['source_file'].isin(
                        st.session_state.filter_sources
                    )
                ]

            # Apply step filter (check which type is active)
            if 'step' in temp_filtered_df.columns:
                # Check range slider state first
                if st.session_state.filter_step_range and isinstance(
                    st.session_state.filter_step_range, tuple
                ):
                    min_s, max_s = st.session_state.filter_step_range
                    # Check if the slider is actually active (not default full range)
                    valid_steps_for_range = (
                        base_df_for_filters['step'].dropna().astype(int)
                    )
                    if not valid_steps_for_range.empty:
                        min_step_orig, max_step_orig = int(
                            valid_steps_for_range.min()
                        ), int(valid_steps_for_range.max())
                        if min_s > min_step_orig or max_s < max_step_orig:
                            temp_filtered_df = temp_filtered_df[
                                temp_filtered_df['step'].between(
                                    min_s, max_s, inclusive='both'
                                )
                            ]

                # Check multiselect state if range wasn't active or doesn't exist
                elif st.session_state.filter_steps_select:
                    temp_filtered_df = temp_filtered_df[
                        temp_filtered_df['step'].isin(
                            st.session_state.filter_steps_select
                        )
                    ]

            # Apply reward filter
            if st.session_state.filter_reward_range and isinstance(
                st.session_state.filter_reward_range, tuple
            ):
                min_r, max_r = st.session_state.filter_reward_range
                # Check if the slider is actually active
                valid_rewards_for_range = base_df_for_filters[
                    'accuracy_reward'
                ].dropna()
                if not valid_rewards_for_range.empty:
                    min_rew_orig, max_rew_orig = float(
                        valid_rewards_for_range.min()
                    ), float(valid_rewards_for_range.max())
                    if min_r > min_rew_orig or max_r < max_rew_orig:
                        temp_filtered_df = temp_filtered_df[
                            temp_filtered_df['accuracy_reward'].between(
                                min_r, max_r, inclusive='both'
                            )
                        ]

            # Apply search filter
            if st.session_state.search_term:
                search_conditions = []
                term = st.session_state.search_term
                if (
                    st.session_state.search_prompt
                    and 'prompt_text' in temp_filtered_df.columns
                ):
                    search_conditions.append(
                        temp_filtered_df['prompt_text']
                        .astype(str)
                        .str.contains(term, case=False, na=False, regex=False)
                    )
                if (
                    st.session_state.search_completion
                    and 'completion_text' in temp_filtered_df.columns
                ):
                    search_conditions.append(
                        temp_filtered_df['completion_text']
                        .astype(str)
                        .str.contains(term, case=False, na=False, regex=False)
                    )
                if search_conditions:
                    temp_filtered_df = temp_filtered_df[
                        pd.concat(search_conditions, axis=1).any(axis=1)
                    ]

            # Update session state
            st.session_state.filtered_df = temp_filtered_df

            # Reset page/selection ONLY if filters were changed by UI interaction (not just initial load)
            # And not if the view button was just clicked
            if filters_changed and not st.session_state.get(
                'view_button_just_clicked', False
            ):
                st.session_state.current_page = 1
                st.session_state.selected_row_index = None
                st.session_state.show_details = False

        # Reset the flag now that filtering is done
        st.session_state.filters_changed_flag = False

    # --- Post-Filtering State Management ---
    # ALWAYS reset the view button flag after checking it, regardless of filter changes
    if st.session_state.get('view_button_just_clicked', False):
        st.session_state.view_button_just_clicked = False

    # Display filter status and reset button
    filtered_df_display = (
        st.session_state.filtered_df
    )  # Use the potentially updated df
    st.sidebar.write(f"Filtered samples: {len(filtered_df_display)}")
    if filters_applied_ui:  # Show reset button if any filter UI is active
        if st.sidebar.button('Reset All Filters'):
            # Clear filter states
            st.session_state.filter_sources = []
            st.session_state.filter_step_range = None
            st.session_state.filter_steps_select = []
            st.session_state.filter_reward_range = None
            st.session_state.search_term = ''
            st.session_state.search_prompt = True
            st.session_state.search_completion = True
            # Force re-filtering with defaults on next run
            st.session_state.filters_changed_flag = True
            # Reset view state
            st.session_state.current_page = 1
            st.session_state.selected_row_index = None
            st.session_state.show_details = False
            st.rerun()

# --- Main Content ---
st.title('LLM Sample Analyzer')

if st.session_state.data_loaded and st.session_state.filtered_df is not None:
    # Use the filtered dataframe from session state
    filtered_df = st.session_state.filtered_df

    tab1, tab2 = st.tabs(['Samples Overview', 'Analytics'])

    with tab1:
        # --- Layout Definition ---
        if (
            st.session_state.show_details
            and st.session_state.selected_row_index is not None
        ):
            col_table, col_details = st.columns([1, 1])
        else:
            col_table = st
            col_details = None

        # --- Table View (col_table) using Loop and Columns ---
        col_table.subheader('Samples')
        if not filtered_df.empty:
            # Pagination Controls
            col_table.select_slider(
                'Rows per page', [5, 10, 20, 50, 100], key='rows_per_page'
            )
            rows_per_page = st.session_state.rows_per_page
            max_pages = max(
                1, (len(filtered_df) + rows_per_page - 1) // rows_per_page
            )
            # Ensure current page is valid after filtering
            if st.session_state.current_page > max_pages:
                st.session_state.current_page = max_pages

            nav_cols = col_table.columns([1, 2, 1])
            if nav_cols[0].button(
                '◀ Prev',
                disabled=(st.session_state.current_page <= 1),
                use_container_width=True,
            ):
                st.session_state.current_page -= 1
                st.rerun()

            page_num_input = nav_cols[1].number_input(
                f"Page (1-{max_pages})",
                min_value=1,
                max_value=max_pages,
                value=st.session_state.current_page,
                key='page_num_input',
                label_visibility='collapsed',
            )
            if page_num_input != st.session_state.current_page:
                st.session_state.current_page = page_num_input
                st.rerun()

            nav_cols[1].write(
                f"Page **{st.session_state.current_page}** of **{max_pages}**"
            )

            if nav_cols[2].button(
                'Next ▶',
                disabled=(st.session_state.current_page >= max_pages),
                use_container_width=True,
            ):
                st.session_state.current_page += 1
                st.rerun()

            start_idx = (st.session_state.current_page - 1) * rows_per_page
            end_idx = min(start_idx + rows_per_page, len(filtered_df))
            col_table.write(
                f"Showing samples {start_idx+1} - {end_idx} of {len(filtered_df)}"
            )

            # Prepare page data
            page_df = filtered_df.iloc[start_idx:end_idx]

            # Display rows using columns for better structure
            for i, (idx, row) in enumerate(page_df.iterrows()):
                with col_table.container():
                    st.markdown('---', unsafe_allow_html=True)
                    row_cols = st.columns([4, 1])

                    # --- Left Column (Info & Previews) ---
                    info_parts = [f"[ Index {idx} ]"]
                    if 'step' in row and pd.notna(row['step']):
                        info_parts.append(f"Step: {int(row['step'])}")
                    if 'accuracy_reward' in row and pd.notna(
                        row['accuracy_reward']
                    ):
                        info_parts.append(
                            f"Accuracy reward: {row['accuracy_reward']:.2f}"
                        )
                    if 'prompt_length' in row and pd.notna(
                        row['prompt_length']
                    ):
                        info_parts.append(
                            f"Prompt Length: {int(row['prompt_length'])}"
                        )
                    if 'completion_length' in row and pd.notna(
                        row['completion_length']
                    ):
                        info_parts.append(
                            f"Completion Length: {int(row['completion_length'])}"
                        )
                    row_cols[0].markdown(
                        f"<div class='sample-info'>{' | '.join(info_parts)}</div>",
                        unsafe_allow_html=True,
                    )

                    # --- Prompt Preview (using truncate_middle) ---
                    prompt_text = row.get('prompt_text', None)
                    if pd.notna(prompt_text):
                        preview = truncate_middle(prompt_text, max_length=250)
                        row_cols[0].markdown(
                            f"<div class='preview-label'>Prompt:</div><div class='sample-prompt-preview'>{preview}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        row_cols[0].markdown(
                            "<div class='preview-label'>Prompt:</div><div class='sample-prompt-preview'>*No prompt text*</div>",
                            unsafe_allow_html=True,
                        )

                    # --- Completion Preview (using truncate_middle) ---
                    completion_text = row.get('completion_text', None)
                    if pd.notna(completion_text):
                        preview = truncate_middle(
                            completion_text, max_length=250
                        )
                        row_cols[0].markdown(
                            f"<div class='preview-label'>Completion:</div><div class='sample-completion-preview'>{preview}</div>",
                            unsafe_allow_html=True,
                        )
                    else:
                        row_cols[0].markdown(
                            "<div class='preview-label'>Completion:</div><div class='sample-completion-preview'>*No completion text*</div>",
                            unsafe_allow_html=True,
                        )

                    # --- Right Column (Button) ---
                    button_key = f"view_details_{idx}"
                    if row_cols[1].button(
                        'View Details',
                        key=button_key,
                        help=f"View details for sample index {idx}",
                    ):
                        st.session_state.selected_row_index = idx
                        st.session_state.show_details = True
                        st.session_state.view_button_just_clicked = True
                        # No need to set filters_changed_flag here
                        st.rerun()  # Rerun will now be faster as filters aren't reapplied

            col_table.markdown('---', unsafe_allow_html=True)

            # Export Button
            csv_data = filtered_df.to_csv(index=False).encode('utf-8')
            col_table.download_button(
                label='Export Filtered Data to CSV',
                data=csv_data,
                file_name=f"filtered_samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime='text/csv',
                key='export_csv',
            )
        else:
            col_table.warning('No samples match the current filters.', icon='⚠️')

        # --- Details Panel (col_details) ---
        if (
            col_details is not None
            and st.session_state.selected_row_index is not None
            and st.session_state.selected_row_index
            in filtered_df.index  # Check index exists
        ):
            try:
                sample = filtered_df.loc[st.session_state.selected_row_index]
                # Find position within the *current* filtered dataframe for nav
                filtered_indices = filtered_df.index.tolist()
                try:
                    sample_idx_pos = filtered_indices.index(
                        st.session_state.selected_row_index
                    )
                except ValueError:
                    # Should not happen if index check above passed, but safety first
                    st.error('Selected index inconsistency.')
                    st.session_state.show_details = False
                    st.session_state.selected_row_index = None
                    st.rerun()

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
                    st.rerun()

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
                    status_class = (
                        'status-good'
                        if acc_val > 0.75
                        else (
                            'status-bad' if acc_val < 0.25 else 'status-neutral'
                        )
                    )
                    gt_acc_cols[1].markdown(
                        f"<div class='status-base {status_class}'>{acc_val:.3f}</div>",
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
                    exclude_cols = [
                        'prompt_text',
                        'completion_text',
                        'prompt_preview',
                        'completion_preview',
                        'details',
                        'index',
                    ]
                    items_found = False
                    for key, value in sample.items():
                        if key not in exclude_cols and pd.notna(value):
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
                nav_cols = col_details.columns(2)
                prev_disabled = sample_idx_pos <= 0
                if nav_cols[0].button(
                    '⬅ Previous Sample',
                    disabled=prev_disabled,
                    use_container_width=True,
                    key='prev_sample',
                ):
                    if not prev_disabled:
                        st.session_state.selected_row_index = filtered_indices[
                            sample_idx_pos - 1
                        ]
                        st.session_state.view_button_just_clicked = True
                        st.rerun()
                next_disabled = sample_idx_pos >= len(filtered_indices) - 1
                if nav_cols[1].button(
                    'Next Sample ➡',
                    disabled=next_disabled,
                    use_container_width=True,
                    key='next_sample',
                ):
                    if not next_disabled:
                        st.session_state.selected_row_index = filtered_indices[
                            sample_idx_pos + 1
                        ]
                        st.session_state.view_button_just_clicked = True
                        st.rerun()

            except KeyError:
                col_details.error(
                    f"Error: Sample index {st.session_state.selected_row_index} not found in the filtered data.",
                    icon='🚨',
                )
                st.session_state.show_details = False
                st.session_state.selected_row_index = None
            except Exception as e:
                col_details.error(f"Error displaying details: {e}", icon='🚨')
                # Optionally reset state on unexpected error
                # st.session_state.show_details = False
                # st.session_state.selected_row_index = None

        elif (
            st.session_state.show_details
            and st.session_state.selected_row_index is not None
        ):
            # Handle case where index was selected but is no longer in the filtered_df
            col_details.warning(
                f"Sample index {st.session_state.selected_row_index} is no longer present with the current filters.",
                icon='⚠️',
            )
            if col_details.button('Close Details'):
                st.session_state.show_details = False
                st.session_state.selected_row_index = None
                st.rerun()

    # --- Analytics Tab ---
    with tab2:
        st.subheader('Data Analysis')
        if not filtered_df.empty:
            # File comparison
            if (
                'source_file' in filtered_df.columns
                and filtered_df['source_file'].nunique() > 1
            ):
                st.markdown('#### Files Overview')
                file_counts = (
                    filtered_df['source_file'].value_counts().reset_index()
                )
                file_counts.columns = ['Source File', 'Sample Count']
                try:
                    fig = px.bar(
                        file_counts,
                        x='Source File',
                        y='Sample Count',
                        title='Samples per Source File',
                        text='Sample Count',
                    )
                    fig.update_layout(xaxis_title=None)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not plot file overview: {e}", icon='⚠️')

            # Scatter plots
            vis_cols = st.columns(2)
            plot_df = filtered_df
            MAX_POINTS_FOR_SCATTER = 2000
            if len(plot_df) > MAX_POINTS_FOR_SCATTER:
                st.info(
                    f"Plotting a random sample of {MAX_POINTS_FOR_SCATTER} points for performance."
                )
                plot_df = plot_df.sample(
                    MAX_POINTS_FOR_SCATTER, random_state=42
                )

            with vis_cols[0]:
                if (
                    'step' in plot_df.columns
                    and not plot_df['step'].isnull().all()
                    and 'accuracy_reward' in plot_df.columns
                    and not plot_df['accuracy_reward'].isnull().all()
                ):
                    st.markdown('#### Accuracy Reward vs. Step')
                    try:
                        fig = px.scatter(
                            plot_df.dropna(subset=['step', 'accuracy_reward']),
                            x='step',
                            y='accuracy_reward',
                            title='Accuracy Reward vs. Training Step',
                            labels={
                                'step': 'Step',
                                'accuracy_reward': 'Accuracy Reward',
                            },
                            hover_data=[
                                (
                                    plot_df.index.name
                                    if plot_df.index.name
                                    else 'index'
                                )
                            ],
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(
                            f"Could not plot accuracy vs step: {e}", icon='⚠️'
                        )
                else:
                    st.markdown(
                        "_(Requires 'step' and 'accuracy_reward' columns)_"
                    )

            with vis_cols[1]:
                if (
                    'prompt_length' in plot_df.columns
                    and not plot_df['prompt_length'].isnull().all()
                    and 'completion_length' in plot_df.columns
                    and not plot_df['completion_length'].isnull().all()
                ):
                    st.markdown('#### Prompt vs. Completion Length')
                    try:
                        fig = px.scatter(
                            plot_df.dropna(
                                subset=['prompt_length', 'completion_length']
                            ),
                            x='prompt_length',
                            y='completion_length',
                            title='Prompt vs. Completion Length',
                            labels={
                                'prompt_length': 'Prompt Length',
                                'completion_length': 'Completion Length',
                            },
                            hover_data=[
                                (
                                    plot_df.index.name
                                    if plot_df.index.name
                                    else 'index'
                                )
                            ],
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(
                            f"Could not plot length vs length: {e}", icon='⚠️'
                        )
                else:
                    st.markdown(
                        "_(Requires 'prompt_length' and 'completion_length' columns)_"
                    )

            # Distribution plots
            dist_cols = st.columns(3)
            with dist_cols[0]:
                if (
                    'accuracy_reward' in filtered_df.columns
                    and not filtered_df['accuracy_reward'].isnull().all()
                ):
                    st.markdown('#### Reward Distribution')
                    try:
                        fig = px.histogram(
                            filtered_df.dropna(subset=['accuracy_reward']),
                            x='accuracy_reward',
                            nbins=30,
                            title='Accuracy Reward Distribution',
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(
                            f"Could not plot reward distribution: {e}", icon='⚠️'
                        )
                else:
                    st.markdown("_(Requires 'accuracy_reward')_")
            with dist_cols[1]:
                if (
                    'prompt_length' in filtered_df.columns
                    and not filtered_df['prompt_length'].isnull().all()
                ):
                    st.markdown('#### Prompt Length')
                    try:
                        fig = px.histogram(
                            filtered_df.dropna(subset=['prompt_length']),
                            x='prompt_length',
                            nbins=30,
                            title='Prompt Length Distribution',
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(
                            f"Could not plot prompt length: {e}", icon='⚠️'
                        )
                else:
                    st.markdown("_(Requires 'prompt_length')_")
            with dist_cols[2]:
                if (
                    'completion_length' in filtered_df.columns
                    and not filtered_df['completion_length'].isnull().all()
                ):
                    st.markdown('#### Completion Length')
                    try:
                        fig = px.histogram(
                            filtered_df.dropna(subset=['completion_length']),
                            x='completion_length',
                            nbins=30,
                            title='Completion Length Distribution',
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.warning(
                            f"Could not plot completion length: {e}", icon='⚠️'
                        )
                else:
                    st.markdown("_(Requires 'completion_length')_")

            # Trends over steps
            if (
                'step' in filtered_df.columns
                and not filtered_df['step'].isnull().all()
                and filtered_df['step'].nunique() > 1
            ):
                st.markdown('#### Trends Over Training Steps')
                try:
                    step_data = filtered_df.dropna(subset=['step']).copy()
                    step_data['step'] = pd.to_numeric(step_data['step'])
                    aggregations = {}
                    if (
                        'accuracy_reward' in step_data.columns
                        and not step_data['accuracy_reward'].isnull().all()
                    ):
                        aggregations['accuracy_reward_mean'] = (
                            'accuracy_reward',
                            'mean',
                        )
                        aggregations['accuracy_reward_std'] = (
                            'accuracy_reward',
                            'std',
                        )
                    if (
                        'prompt_length' in step_data.columns
                        and not step_data['prompt_length'].isnull().all()
                    ):
                        aggregations['prompt_length_mean'] = (
                            'prompt_length',
                            'mean',
                        )
                    if (
                        'completion_length' in step_data.columns
                        and not step_data['completion_length'].isnull().all()
                    ):
                        aggregations['completion_length_mean'] = (
                            'completion_length',
                            'mean',
                        )

                    if aggregations:
                        agg_results = (
                            step_data.groupby('step')
                            .agg(**aggregations)
                            .reset_index()
                        )
                        if 'accuracy_reward_mean' in agg_results.columns:
                            st.markdown('##### Mean Accuracy Reward')
                            fig = px.line(
                                agg_results,
                                x='step',
                                y='accuracy_reward_mean',
                                title='Avg Accuracy Reward / Step',
                                labels={
                                    'step': 'Step',
                                    'accuracy_reward_mean': 'Mean Reward',
                                },
                                error_y=(
                                    'accuracy_reward_std'
                                    if 'accuracy_reward_std'
                                    in agg_results.columns
                                    else None
                                ),
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        len_cols = st.columns(2)
                        with len_cols[0]:
                            if 'prompt_length_mean' in agg_results.columns:
                                st.markdown('##### Mean Prompt Length')
                                fig = px.line(
                                    agg_results,
                                    x='step',
                                    y='prompt_length_mean',
                                    title='Avg Prompt Length / Step',
                                    labels={
                                        'step': 'Step',
                                        'prompt_length_mean': 'Mean Length',
                                    },
                                )
                                st.plotly_chart(fig, use_container_width=True)
                        with len_cols[1]:
                            if 'completion_length_mean' in agg_results.columns:
                                st.markdown('##### Mean Completion Length')
                                fig = px.line(
                                    agg_results,
                                    x='step',
                                    y='completion_length_mean',
                                    title='Avg Completion Length / Step',
                                    labels={
                                        'step': 'Step',
                                        'completion_length_mean': 'Mean Length',
                                    },
                                )
                                st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating trends: {e}", icon='🚨')
            elif (
                'step' in filtered_df.columns
                and filtered_df['step'].nunique() <= 1
            ):
                st.info(
                    'Trend analysis requires data from multiple training steps.'
                )
        else:
            st.warning('No samples to analyze with current filters.', icon='⚠️')

# --- Initial State / No Data Loaded ---
elif not selected_files:
    st.info('Select data source and files from the sidebar to begin analysis.')
    # (How to Use section remains the same)
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
         "source_file": "optional_filename.jsonl"
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
else:  # Handles the case where data loading failed after files were selected
    st.error(
        'Data could not be loaded. Please check file paths and formats in the sidebar.',
        icon='🚨',
    )
