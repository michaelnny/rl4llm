"""RL actor to collect samples"""

import logging
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from rl4llm.utils import TrainingTracker
from rl4llm.envs import VectorEnvWrapper
from rl4llm.generations import LLMGenerator, OpenAIGenerator
from rl4llm.constants import DEFAULT_FAILED_RESPONSE


from rl4llm.types import DecodingConfig, EnvAction, EnvState, Episode, ExplorationConfig
from rl4llm.utils import clean_up_gpu_memory, is_texts_similar

logger = logging.getLogger()


class EgreedyActor:
    """Epsilon-greedy actor for generating responses with language model."""

    def __init__(
        self,
        generator: Union[LLMGenerator, OpenAIGenerator],
        decoding_config: DecodingConfig,
        exploration_config: Optional[ExplorationConfig] = None,
        tracker: Optional[TrainingTracker] = None,
        for_evaluator: Optional[bool] = False,
    ):
        """Initialize the actor."""
        assert isinstance(decoding_config, DecodingConfig), 'Invalid decoding configuration'

        self.generator = generator
        self.decoding_config = decoding_config
        self.exploration_config = exploration_config
        self.is_oai_generator = isinstance(generator, OpenAIGenerator)

        self.tracker = tracker
        # self.repetition_detector = RepetitionDetector(ngram_repeat_threshold=8, sentence_repeat_threshold=3)

        self.for_evaluator = for_evaluator
        self.role_name = 'Evaluator' if for_evaluator else 'Actor'
        self._reset_counters()

    def _reset_counters(self) -> None:
        """Reset internal counters"""
        self.step_count = 0
        self.episode_count = 0
        self.iter_count = 0

    def get_epsilon(self) -> float:
        """Compute current epsilon value based on episode count."""
        if self.for_evaluator or self.is_oai_generator:
            return 0.0
        if not self.exploration_config or not isinstance(self.exploration_config, ExplorationConfig):
            return 0.0
        if self.exploration_config.decay_steps <= 0:
            return self.epsilon

        decay_rate = (
            self.exploration_config.init_epsilon - self.exploration_config.min_epsilon
        ) / self.exploration_config.decay_steps
        self.epsilon = max(
            self.exploration_config.min_epsilon, self.exploration_config.init_epsilon - decay_rate * self.episode_count
        )
        return self.epsilon

    def _should_do_exploring_start(self) -> bool:
        """Determine if exploration should be performed."""
        if self.for_evaluator or self.is_oai_generator:
            return False
        return np.random.rand() < self.get_epsilon()

    def act(self, states: List[EnvState]) -> List[EnvAction]:
        """Generate responses for given states, handling None states."""
        # Filter out None states
        valid_states = [s for s in states if s is not None]

        if not valid_states:
            return [None] * len(states)

        exploring_steps = (
            random.randint(1, self.exploration_config.max_explore_steps) if self._should_do_exploring_start() else 0
        )

        rollouts = self.generator.generate_actions_for_rl(
            batch_states=valid_states,
            max_new_tokens=self.decoding_config.max_new_tokens,
            temperature=self.decoding_config.temperature,
            top_p=self.decoding_config.top_p,
            top_k=self.decoding_config.top_k,
            exploring_steps=exploring_steps,
        )

        # Map the actions back to the original state positions
        actions = []
        rollout_idx = 0
        for state in states:
            if state is not None:
                actions.append(self._process_rollouts([rollouts[rollout_idx]])[0])
                rollout_idx += 1
            else:
                actions.append(None)

        self.step_count += len(valid_states)
        return actions

    def generate_samples(
        self,
        vector_env: VectorEnvWrapper,
        max_episodes: int,
        correct_answer_rate: float = 0.0,
        strict_duplicate: bool = False,
    ) -> Tuple[List[Episode], Dict]:
        """Generate samples ensuring minimum correct answer rate."""
        assert 0.0 <= correct_answer_rate <= 1.0, 'correct_answer_rate must be between 0 and 1'
        clean_up_gpu_memory()

        stats_tracker = self._init_stats_tracker()
        collected_episodes = []
        question_cache = {}

        states = vector_env.reset()

        with tqdm(total=max_episodes, desc=f'{self.role_name} generating episodes', unit='episode') as pbar:
            while len(collected_episodes) < max_episodes:
                # Identify active environments and their states
                active_env_indices = [i for i, state in enumerate(states) if state is not None]
                active_states = [states[i] for i in active_env_indices]

                # Generate actions only for active environments
                actions = [None] * vector_env.num_envs  # Initialize with None
                if active_states:
                    active_actions = self.act(active_states)
                    for i, action in zip(active_env_indices, active_actions):
                        actions[i] = action

                # Take step in all environments (None actions for finished ones)
                new_states = vector_env.step(actions)

                # Process completed episodes and reset finished environments
                for i in range(vector_env.num_envs):
                    if states[i] is not None and vector_env.is_done(i):  # Check if was active and is now done
                        episode = vector_env.get_episode(i)
                        processed = self._process_batch_episodes(
                            [episode],
                            question_cache,
                            collected_episodes,
                            correct_answer_rate,
                            vector_env.max_reward,
                            stats_tracker,
                            strict_duplicate,
                        )

                        if processed:
                            collected_episodes.extend(processed)
                            pbar.update(1)

                        # Reset the environment and update the state
                        new_states[i] = vector_env.reset_one(i)

                states = new_states

                # Save samples periodically
                if self.tracker and len(collected_episodes) and len(collected_episodes) % 50 == 0:
                    self.tracker.flush()

        iter_stats = self._compute_iteration_stats(
            stats_tracker, collected_episodes, vector_env.max_reward, pbar.format_dict.get('elapsed', 0)
        )

        self._log_iteration_results(iter_stats)
        return collected_episodes, iter_stats

    def _init_stats_tracker(self) -> Dict:
        """Initialize statistics tracking dictionary"""
        return {'total': 0, 'correct': 0, 'bad': 0, 'duplicate': 0}

    def _process_batch_episodes(
        self,
        batch_episodes: List[Episode],
        question_cache: Dict[str, List[Episode]],
        collected_episodes: List[Episode],
        correct_answer_rate: float,
        max_reward: float,
        stats_tracker: Dict,
        strict_duplicate: bool,
    ) -> List[Episode]:
        """Process and filter batch episodes"""
        current_batch_size = len(batch_episodes)
        stats_tracker['total'] += current_batch_size

        if not self.for_evaluator:
            filtered_episodes = self._filter_out_bad_samples(batch_episodes)
            # Calculate bad count based on current batch only
            bad_count = current_batch_size - len(filtered_episodes)
            stats_tracker['bad'] += bad_count
            batch_episodes = filtered_episodes

        processed_episodes = []
        for episode in batch_episodes:
            if self._is_duplicate_episode(episode, question_cache, strict_duplicate):
                stats_tracker['duplicate'] += 1
                continue

            if self._should_collect_episode(episode, collected_episodes, correct_answer_rate, max_reward):
                processed_episodes.append(episode)
                self._update_question_cache(episode, question_cache)
                self.episode_count += 1
                self._log_episode(episode)

        return processed_episodes

    def _process_rollouts(
        self,
        actions: List[EnvAction],
    ) -> List[EnvAction]:
        """Process and validate rollout actions"""

        for action in actions:
            if not action.text:
                action.text = DEFAULT_FAILED_RESPONSE

        return actions

    def _filter_out_bad_samples(self, samples: List[Episode]) -> List[Episode]:
        """Filter out invalid or low-quality samples"""
        return [sample for sample in samples if self._is_valid_sample(sample)]

    def _is_valid_sample(self, sample: Episode) -> bool:
        """Check if a sample meets quality criteria"""
        if sample.count_completion_tokens() < 50:
            logger.debug('Skip too short episode')
            return False

        # could be 'dummy' response
        if any(
            [
                True
                for t in sample.transitions
                if len(t.action.text) < 50 or DEFAULT_FAILED_RESPONSE.lower() in t.action.text.lower()
            ]
        ):
            logger.debug('Skip too short transition')
            return False

        # for t in sample.transitions:
        #     has_repetition, results = self.repetition_detector.analyze_text(t.action.text, n=0)  # n=0 don't check n-grams
        #     if has_repetition:
        #         logger.debug(f"Skip repetition episode {results}")
        #         return False

        return True

    def _is_duplicate_episode(self, episode: Episode, question_cache: Dict[str, List[Episode]], strict_duplicate: bool) -> bool:
        """Check if episode is a duplicate"""
        if episode.question not in question_cache:
            return False

        if strict_duplicate and episode.question in question_cache:
            return True

        # only compare answers
        return any(
            is_texts_similar(episode.transitions[-1].action.text, prev_ep.transitions[-1].action.text, 0.9)
            for prev_ep in question_cache[episode.question]
        )

    def _update_question_cache(self, episode: Episode, question_cache: Dict[str, List[Episode]]) -> None:
        """Update question cache with new episode"""
        if episode.question not in question_cache:
            question_cache[episode.question] = []
        question_cache[episode.question].append(episode)

    def _should_collect_episode(
        self, episode: Episode, collected_episodes: List[Episode], correct_answer_rate: float, max_reward: float
    ) -> bool:
        """Determine if episode should be collected"""
        is_correct = episode.graded_reward == max_reward

        # If correct_answer_rate is 1.0, only collect correct samples
        if correct_answer_rate == 1.0:
            return is_correct

        if is_correct or self.for_evaluator:
            return True

        current_total = len(collected_episodes)
        if current_total == 0:
            return True

        current_correct = sum([1 for ep in collected_episodes if ep.graded_reward == max_reward])
        # Check if collecting this episode would maintain the desired correct_answer_rate
        return (current_correct / (current_total + 1)) >= correct_answer_rate

    def _log_episode(self, episode: Episode) -> None:
        """Log episode information if tracker is available"""
        if self.tracker:
            self.tracker.log_actor_step(episode, self.for_evaluator)

    def _compute_iteration_stats(
        self, stats_tracker: Dict, collected_episodes: List[Episode], max_reward: float, elapsed_time: float
    ) -> Dict:
        """Compute comprehensive iteration statistics"""

        episode_prompt_tokens = np.array([ep.count_prompt_tokens() for ep in collected_episodes])
        episode_completion_tokens = np.array([ep.count_completion_tokens() for ep in collected_episodes])
        episode_total_tokens = np.array([ep.count_total_tokens() for ep in collected_episodes])
        correct_samples = sum(1 for ep in collected_episodes if ep.graded_reward == max_reward)

        base_stats = {
            'elapsed/time': round(elapsed_time, 4),
            'elapsed/step_time': round(elapsed_time / max(stats_tracker['total'], 1), 4),
            'elapsed/total_episodes': stats_tracker['total'],
            'exploration_epsilon': self.get_epsilon(),
            'objective/accuracy': correct_samples / stats_tracker['total'],
            'objective/correct_ratio': correct_samples / len(collected_episodes),
            'objective/bad_ratio': stats_tracker['bad'] / stats_tracker['total'],
            'objective/duplicate_ratio': stats_tracker['duplicate'] / stats_tracker['total'],
            'episode/completion_tokens': np.mean(episode_completion_tokens).item(),
            'episode/completion_tokens_p90': np.percentile(episode_completion_tokens, 90).item(),
            'usage/total_tokens': np.sum(episode_total_tokens),
            'usage/prompt_tokens': np.sum(episode_prompt_tokens),
            'usage/completion_tokens': np.sum(episode_completion_tokens),
        }

        step_stats = self._compute_step_level_stats(collected_episodes)
        return {**base_stats, **step_stats}

    @staticmethod
    def _compute_step_level_stats(collected_episodes: List[Episode]) -> Dict:
        """Compute detailed statistics for each step"""
        steps_data = {}

        for episode in collected_episodes:
            for t in episode.transitions:
                step_name = t.state.state_id
                if step_name not in steps_data:
                    steps_data[step_name] = {'lengths': [], 'rewards': []}

                steps_data[step_name]['rewards'].append(t.reward)

                try:
                    steps_data[step_name]['lengths'].append(t.action.usage.completion_tokens)
                except Exception as _e:
                    logger.error(f"Error processing step {step_name}: {_e}")
                    steps_data[step_name]['lengths'].append(0)

        stats = {}
        for step_name, data in steps_data.items():
            if not data['lengths']:
                continue

            for metric_name, values in [('length', data['lengths']), ('reward', data['rewards'])]:
                array_data = np.array(values)
                base_key = f"episode/{step_name}_{metric_name}"
                stats[f"{base_key}_mean"] = float(np.mean(array_data))
                stats[f"{base_key}_std"] = float(np.std(array_data))

        return stats

    def _log_iteration_results(self, iter_stats: Dict) -> None:
        """Log iteration results and update tracker"""
        self.iter_count += 1
        logger.info(f"{self.role_name} stats: {iter_stats}")

        if self.tracker:
            self.tracker.log_actor_iteration_stats(iter_stats, self.for_evaluator)
