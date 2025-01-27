import random
from typing import Dict, List, Union

import numpy as np
from datasets import Dataset, concatenate_datasets

from rl4llm.envs import MDPEnv
from rl4llm.types import EnvAction, EnvState, Episode


class VectorEnvWrapper:
    def __init__(
        self,
        datasets: Union[Dataset, List[Dataset]],
        num_envs: int = 4,
        shuffle: bool = True,
        **kwargs,
    ):
        """
        Initialize vectorized environments with distinct, optionally shuffled subsets.

        Args:
            datasets (Dataset or List[Dataset]):
                - Single dataset or list of datasets to be split among environments
            num_envs (int): Number of parallel environments.
            shuffle (bool): Whether to shuffle the dataset before splitting.
            **kwargs: Additional keyword arguments to pass to each MDPEnv.
        """

        # Ensure datasets is a list, even if a single dataset is provided
        if not isinstance(datasets, list):
            datasets = [datasets]

        # Validate inputs
        assert num_envs >= 1, 'Number of environments must be at least 1'
        assert len(datasets) > 0, 'At least one dataset must be provided'

        self.num_envs = num_envs
        self.shuffle = shuffle
        self.shared_kwargs = kwargs

        # Set seed for reproducibility
        self.seed = kwargs.get('seed', 42)
        np.random.seed(self.seed)
        random.seed(self.seed)  # If using random.shuffle

        self.subsets = self._prepare_datasets(datasets)

        # Validation: Ensure num_envs does not exceed total samples
        total_samples = sum(len(subset) for subset in self.subsets)
        assert (
            self.num_envs <= total_samples
        ), f"Number of environments ({self.num_envs}) cannot exceed the total number of samples ({total_samples})"

        # Initialize environments with their respective subsets and unique kwargs
        self.envs: List[MDPEnv] = []
        for i in range(num_envs):
            env_kwargs = self.shared_kwargs.copy()
            # Ensure unique seed per environment if 'seed' is provided
            env_kwargs['seed'] = self.seed + i
            # Initialize the MDPEnv with its subset and kwargs
            env = MDPEnv(dataset=self.subsets[i], **env_kwargs)
            self.envs.append(env)

        # Track episodes per environment
        self.min_reward = self.envs[0].min_reward
        self.max_reward = self.envs[0].max_reward

    def _prepare_datasets(self, datasets: List[Dataset]) -> List[List[dict]]:
        """
        Prepare and split multiple datasets across environments.

        Args:
            datasets (List[Dataset]): List of datasets to process

        Returns:
            List of subsets, where each subset is a list of data samples
        """
        # First, concatenate/merge all datasets
        merged_dataset = concatenate_datasets(datasets)

        # Shuffle if required
        if self.shuffle:
            merged_dataset = merged_dataset.shuffle(seed=self.seed)

        # Split into num_envs subsets while maintaining Dataset type
        return self._split_dataset(merged_dataset, self.num_envs)

    def _split_dataset(self, dataset: Dataset, num_splits: int) -> List[Dataset]:
        """
        Split the dataset into num_splits approximately equal subsets.

        Args:
        dataset (Dataset): The dataset to split
        num_splits (int): Number of splits

        Returns:
        List[Dataset]: A list containing num_splits Dataset subsets
        """
        # Calculate cumulative indices for splitting
        total_size = len(dataset)
        split_indices = [0]
        current_index = 0

        # Calculate split points
        for i in range(num_splits):
            # Calculate the size for this split
            split_size = total_size // num_splits
            if i < total_size % num_splits:
                split_size += 1

            current_index += split_size
            split_indices.append(current_index)

        # Create subsets using the indices
        subsets = []
        for i in range(num_splits):
            start = split_indices[i]
            end = split_indices[i + 1]
            subset = dataset.select(range(start, end))
            subsets.append(subset)

        return subsets

    def step(self, actions: List[EnvAction]) -> List[EnvState]:
        """
        Step all environments with the given actions.

        Args:
            actions (List[EnvAction]): Actions for each environment (None for finished envs).

        Returns:
            List[EnvState]: A list of next states for each environment.
        """
        next_states = []
        for i, env in enumerate(self.envs):
            if not self._done_flags[i]:  # Only step if not done
                next_state = env.step(actions[i])
                if env.is_done():
                    self._done_flags[i] = True
                next_states.append(next_state)
            else:
                next_states.append(None)  # Keep None state for finished envs
        return next_states

    def is_done(self, env_idx: int) -> bool:
        """
        Check if a specific environment is done.

        Returns:
            bool: indicating done status.
        """
        return self._done_flags[env_idx]

    def reset_one(self, env_idx: int) -> EnvState:
        """
        Reset a specific environment and return its initial state.

        Args:
            env_idx (int): Index of the environment to reset

        Returns:
            EnvState: Initial state of the reset environment

        Raises:
            IndexError: If env_idx is out of range
        """
        if not 0 <= env_idx < self.num_envs:
            raise IndexError(f"Environment index {env_idx} out of range [0, {self.num_envs})")
        self._done_flags[env_idx] = False  # Reset done flag
        return self.envs[env_idx].reset()

    def reset(self) -> List[EnvState]:
        """Reset all environments and their done flags."""
        self._done_flags = [False] * self.num_envs
        return [env.reset() for env in self.envs]

    def get_all_episodes(self) -> List[Episode]:
        """
        Get the current episodes from all environments.

        Returns:
            List: List of current episodes from each environment.
        """
        return [env.get_current_episode() for env in self.envs]

    def get_episode(self, env_idx: int) -> Episode:
        """
        Get the current episode from a specific environment.

        Args:
            env_idx (int): Index of the environment to get the episode from

        Returns:
            Episode: The current episode from the specified environment

        Raises:
            IndexError: If env_idx is out of range
        """
        if not 0 <= env_idx < self.num_envs:
            raise IndexError(f"Environment index {env_idx} out of range [0, {self.num_envs})")
        return self.envs[env_idx].get_current_episode()
