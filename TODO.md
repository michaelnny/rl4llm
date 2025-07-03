# TODO: RL4LLM Project Improvement Plan

This document outlines the key areas for improving the RL4LLM framework, focusing on enhancing its robustness, usability, and maintainability for the research community.

## 1. Project Structure & Dependencies

### Findings
- **Inconsistent Dependency Definitions:** The project has both a `requirements.txt` and a `pyproject.toml` file, which define overlapping and slightly different versions of dependencies. This can lead to environment inconsistencies and make dependency management confusing. For example, `deepspeed` is `0.16.1` in `pyproject.toml` but `0.16.7` in `requirements.txt`.
- **Redundant `project.scripts`:** The `pyproject.toml` defines a `[project.scripts]` entry, but the primary way of running experiments is through direct execution of Python scripts in the `scripts/` directory. This entry point seems unused.

### Proposed Actions
- **Consolidate Dependencies:**
    - **Action:** Unify all dependencies into `pyproject.toml`. Remove `requirements.txt` or regenerate it from `pyproject.toml` to ensure they are in sync.
    - **Rationale:** A single source of truth for dependencies simplifies environment setup and improves reproducibility.
    - **Status:** Completed. `requirements.txt` removed, `pyproject.toml` updated.
- **Clarify Project Entry Points:**
    - **Action:** Either remove the `[project.scripts]` entry from `pyproject.toml` or refactor the training scripts to use a single CLI entry point.
    - **Rationale:** Simplifies how users interact with the framework and removes dead code.
    - **Status:** Completed. `[project.scripts]` removed, `scripts/README.md` confirmed to be sufficient.

## 2. Code Quality & Design

### Findings
- **Hardcoded Model/Tokenizer Loading:** The `build_policy_model_and_tokenizer` function in `model_utils.py` is not present in the provided code, but the training script suggests a rigid model loading process. The `apply_custom_chat_template` function in `run_train_grpo.py` is a good feature but is applied globally and not easily configurable.
- **Complex `BaseRLTrainer`:** The `BaseRLTrainer` class is doing too much. It handles model and optimizer configuration, device placement (`_prepare_for_generation`, `_prepare_for_training`), weight synchronization (`synchronize_reference_model`, `synchronize_policy_model`), and the main training loop. This makes it hard to extend and maintain.
- **Manual Memory Management:** The trainer manually moves models to and from the CPU/GPU to save memory. This is complex, error-prone, and tightly coupled with the training loop. A more abstract and robust solution is needed.
- **Inflexible Environment Design:** The `SglMDPEnv` and `SglToolMDPEnv` are good starting points, but they are not easily extensible. The interaction logic is hardcoded within the `_run_interaction_loop` method, making it difficult to implement new interaction patterns without creating a new environment class from scratch.

### Proposed Actions
- **Refactor `BaseRLTrainer`:**
    - **Action:** Decompose `BaseRLTrainer` into smaller, more focused components.
        - Create a `ModelManager` or `DeviceManager` class to handle device placement, model state (train/eval), and memory management.
        - Move weight synchronization logic into a separate `SyncController`.
    - **Rationale:** Improves separation of concerns, making the trainer easier to understand, test, and extend.
    - **Status:** Completed. `DeviceManager` and `SyncController` created and integrated.
- **Improve Environment Extensibility:**
    - **Action:** Introduce a more modular design for environments. Instead of a single `_run_interaction_loop`, break it down into smaller, overridable methods like `prepare_prompts`, `process_completions`, and `is_episode_done`.
    - **Rationale:** Allows researchers to customize specific parts of the interaction loop without rewriting the entire thing.
    - **Status:** Completed. `SglMDPEnv` and `SglToolMDPEnv` refactored.
- **Generalize Chat Template Handling:**
    - **Action:** Move the chat template logic into the environment or a data processing utility, and make it configurable from the YAML file.
    - **Rationale:** Decouples chat formatting from the training script and makes it easier to experiment with different prompt structures.
    - **Status:** Completed. `prompt_template_utils.py` created and integrated into `run_train_grpo.py` and `grpo_config.yaml`.

## 3. Configuration & Usability

### Findings
- **Scattered Configuration:** Configuration is spread across YAML files (`grpo_config.yaml`), command-line arguments in the training scripts, and hardcoded values in the code. For example, the inference server details are passed via CLI args, while most other parameters are in YAML.
- **Lack of Configuration Validation:** The `BaseRLConfig` uses Pydantic, which is great for validation. However, many configurations are still passed around as dictionaries, bypassing this validation.
- **Verbose and Complex Training Scripts:** The `run_train_grpo.py` script contains a lot of boilerplate code for setting up models, tokenizers, and environments. This will be duplicated across all training scripts.

### Proposed Actions
- **Centralize Configuration:**
    - **Action:** Consolidate all configuration into the YAML files. The training scripts should only be responsible for parsing the config file and launching the trainer.
    - **Rationale:** A single, unified configuration file is easier to manage, version, and share.
    - **Status:** Completed. Inference server configuration moved to `grpo_config.yaml` and removed from `run_train_grpo.py` CLI arguments.
- **Create a `TrainerFactory`:**
    - **Action:** Implement a factory pattern to create the trainer, environment, models, and other components based on the configuration file.
    - **Rationale:** Reduces boilerplate code in the training scripts and makes it easier to add new algorithms and environments.
    - **Status:** Completed. `TrainerFactory` created and integrated into `run_train_grpo.py`.
- **Improve Documentation:**
    - **Action:** Add more detailed documentation, especially for the configuration options and the different components of the framework. A "Getting Started" guide would be very helpful.
    - **Rationale:** Makes the framework more accessible to new users.
    - **Status:** Completed. `docs/getting_started.md` created.

## 4. Testing & Reliability

### Findings
- **Limited Test Coverage:** The `tests/` directory exists, but it seems to cover only a fraction of the codebase. Key components like the trainers and environments have limited or no tests.
- **No Integration Tests:** There are no tests that cover the end-to-end workflow of running a training job, from loading data to updating the model.
- **Manual Error Handling:** The code has some `try...except` blocks, but error handling is not systematic. For example, the `synchronize_policy_model` method has complex error handling that could be simplified.

### Proposed Actions
- **Increase Test Coverage:**
    - **Action:** Add unit tests for all major components, including trainers, environments, and reward functions.
    - **Rationale:** Improves code quality and reduces the risk of regressions.
    - **Status:** In Progress. Unit tests for `DeviceManager`, `SyncController`, `prompt_template_utils`, and `TrainerFactory` created.
- **Add Integration Tests:**
    - **Action:** Create integration tests that run a small training job with a mock model and environment.
    - **Rationale:** Ensures that the different components of the framework work together as expected.
    - **Status:** Pending.
- **Implement a CI Pipeline:**
    - **Action:** Set up a Continuous Integration (CI) pipeline using GitHub Actions to automatically run tests on every pull request.
    - **Rationale:** Automates the testing process and helps maintain code quality.
    - **Status:** Pending.
