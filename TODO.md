
# TODO - RL4LLM Project Refinement

This document outlines the plan to transition the RL4LLM project from an MVP to a robust and maintainable tool for the research community.

## 1. Core Abstractions Refactoring

### 1.1. Refactor Trainers to Reduce Code Duplication

**Problem:** Significant code duplication is anticipated across different RL trainer implementations (PPO, GRPO, DAPO).

**Action:**
- [ ] Identify common logic in `ppo_trainer.py`, `grpo_trainer.py`, and other trainer files.
- [ ] Move shared functionalities (e.g., experience collection, data loading, model updates) into the `BaseRLTrainer`.
- [ ] Update the individual trainer classes to inherit and extend the common logic from `BaseRLTrainer`.

### 1.2. Decompose `BaseMDPEnv` for Better Modularity

**Problem:** The `BaseMDPEnv` class is overly complex, handling too many responsibilities.

**Action:**
- [ ] Create a `RewardManager` class to encapsulate all reward-related logic (calculation, transformation).
- [ ] Create a `TokenizationManager` class to handle tokenization, masking, and other token-related operations.
- [ ] Refactor `BaseMDPEnv` to delegate these responsibilities to the new manager classes.

### 1.3. Flexible Chat Templates

**Problem:** The chat template logic is not easily customizable for different models or use cases.

**Action:**
- [ ] Expose a function in `BaseMDPEnv` that allows users to easily implement or override the chat template logic.

## 2. Configuration Management

### 2.1. Unified and Validated Configuration

**Problem:** Configuration is scattered across YAML files and command-line arguments, with no validation.

**Action:**
- [ ] Implement a unified configuration system using Pydantic to validate and merge configurations from all sources.
- [ ] Consolidate the configuration into a single object that is passed to all components.

## 3. Project Structure and CLI

### 3.1. Reorganize Project Layout

**Problem:** The project structure is not well-organized, with files scattered across different folders.

**Action:**
- [ ] Move training scripts from the `scripts` directory to a new `src/rl4llm/cli` directory.
- [ ] Create subdirectories in `src/rl4llm` for different components (e.g., `config`, `data`, `models`, `trainers`).

### 3.2. Refactor Training Scripts

**Problem:** The training scripts contain hardcoded, monolithic logic that is difficult to reuse.

**Action:**
- [ ] Refactor the training scripts to be more modular and reusable by breaking them into smaller functions.
- [ ] Move domain-specific logic (e.g., math problem grader) to separate files.

## 4. Code Quality and Maintainability

### 4.1. Centralize Constants

**Problem:** Hardcoded strings and magic numbers are scattered throughout the codebase.

**Action:**
- [ ] Create a `src/rl4llm/constants.py` file.
- [ ] Move all hardcoded values (e.g., `"accuracy_reward"`, `"assistant"`, role names) to this file.
- [ ] Update the codebase to import and use these constants.

### 4.2. Enhance Documentation

**Problem:** Inline documentation (docstrings, comments) can be improved for clarity.

**Action:**
- [ ] Review and enhance docstrings for all public classes and methods, explaining their purpose, parameters, and return values.
- [ ] Add comments to clarify complex logic, especially in the core abstractions and training loops.

## 5. Testing and Validation

### 5.1. Improve Test Coverage

**Problem:** The extent of test coverage is unclear.

**Action:**
- [ ] Analyze the existing tests in the `tests/` directory.
- [ ] Write new unit tests for the core abstractions (`BaseRLTrainer`, `BaseMDPEnv`, etc.).
- [ ] Add integration tests to verify the interactions between different components (e.g., trainer, environment, inference client).

## 6. User Experience and Onboarding

### 6.1. Improve "Getting Started" Documentation

**Problem:** The current documentation is good, but could be more comprehensive for new users.

**Action:**
- [ ] Create a `docs/getting_started.md` file with a step-by-step guide for setting up the project, running experiments, and implementing custom components.
- [ ] Add a "Project Structure" section to the `README.md` to explain the purpose of each directory.
- [ ] Provide more detailed examples of how to use the framework for different use cases.
