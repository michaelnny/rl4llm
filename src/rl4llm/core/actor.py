"""Actor class for generating samples using the deepspeed inference engine."""

import logging
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from rl4llm.core.base_ds_class import BaseDeepSpeedClass
from rl4llm.envs import VectorEnvWrapper
from rl4llm.types import DecodingConfig, EnvAction, EnvState, Episode, TokenUsage
from rl4llm.utils import TrainingTracker


class Actor(BaseDeepSpeedClass):
    """Implements the actor for generating samples using the deepspeed inference engine."""

    def __init__(
        self,
        config: Dict[str, Any],
        local_rank: int,
        dtype: Optional[torch.dtype] = torch.bfloat16,
        tracker: Optional[TrainingTracker] = None,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(config, local_rank, dtype, tracker, logger)
        self.cpu_model: PreTrainedModel = self._load_inference_model()  # Load model on CPU initially
        self.inference_engine: deepspeed.InferenceEngine = self._create_inference_engine()

    def _load_inference_model(self) -> PreTrainedModel:
        """Loads the causal LM for actor inference."""
        self.logger.info(f'Loading model: {self.pretrained_model_name_or_path}')
        model = AutoModelForCausalLM.from_pretrained(self.pretrained_model_name_or_path, torch_dtype=self.dtype, use_cache=False)
        for param in model.parameters():
            param.requires_grad = False
        return model

    def _create_inference_engine(self) -> deepspeed.InferenceEngine:
        """Creates DeepSpeed inference engine and moves model to device."""
        engine = self._create_deepspeed_inference_engine(self.cpu_model)
        return engine.to(self.device)

    def sync_model_weights(self, model_state_dict: Dict) -> None:
        """Sync the model weights with the learner model."""
        self.logger.info('Syncing model weights with learner')
        self.cpu_model.load_state_dict(model_state_dict, strict=False)
        if hasattr(self, 'inference_engine') and self.inference_engine is not None:
            del self.inference_engine
            torch.cuda.empty_cache()
        self.inference_engine = self._create_inference_engine()

    def offload_for_training(self) -> None:
        """Offload model to CPU for training."""
        self.cpu_model = self.cpu_model.to('cpu')
        del self.inference_engine
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def act(self, batch_states: List[EnvState], decoding: DecodingConfig) -> List[EnvAction]:
        """Generate responses for given states, handling None states."""
        valid_states = [s for s in batch_states if s is not None]
        if not valid_states:
            return [None] * len(batch_states)

        batch_messages = [[{'role': t.role, 'content': t.content} for t in states] for states in valid_states]
        message_prompt = self.tokenizer.apply_chat_template(batch_messages, tokenize=False, add_generation_prompt=True)

        inputs = self.tokenizer(
            message_prompt,
            return_tensors='pt',
            truncation=True,
            padding=True,
            padding_side='left',
            max_length=self.tokenizer.model_max_length,
        ).to(self.device)

        generation_kwargs = {
            'input_ids': inputs.input_ids,
            'attention_mask': inputs.attention_mask,
            'eos_token_id': self.eos_token_id,
            'pad_token_id': self.pad_token_id,
            'use_cache': True,
            'output_scores': True,
            'output_logits': True,
            'return_dict_in_generate': True,
            'return_legacy_cache': False,
            'max_new_tokens': decoding.max_new_tokens,
            'temperature': decoding.temperature if decoding.temperature is not None else 1.0,
            'top_p': decoding.top_p if decoding.top_p is not None else None,
            'top_k': decoding.top_k if decoding.top_k is not None else None,
            'do_sample': decoding.do_sample,
        }

        outputs = self.inference_engine.generate(**generation_kwargs)

        generated_sequences = outputs.sequences
        prompt_length = inputs.input_ids.size(1)
        batch_completion_ids = generated_sequences[:, prompt_length:]
        prompt_tokens_count = (inputs.input_ids != self.pad_token_id).sum(dim=1).cpu().tolist()
        completion_tokens_count = (batch_completion_ids != self.pad_token_id).sum(dim=1).cpu().tolist()
        completion_texts = self.tokenizer.batch_decode(batch_completion_ids, skip_special_tokens=True)

        actions = []
        for i in range(generated_sequences.size(0)):
            actions.append(
                EnvAction(
                    text=completion_texts[i],
                    temperature=generation_kwargs['temperature'],
                    usage=TokenUsage(
                        prompt_tokens=prompt_tokens_count[i],
                        completion_tokens=completion_tokens_count[i],
                        total_tokens=prompt_tokens_count[i] + completion_tokens_count[i],
                    ),
                )
            )

        full_actions = [None] * len(batch_states)  # Distribute actions, handling None states
        action_idx = 0
        for i in range(len(batch_states)):
            if batch_states[i] is not None:
                full_actions[i] = actions[action_idx]
                action_idx += 1
        return full_actions

    def generate_samples(
        self,
        vector_env: VectorEnvWrapper,
        decoding: DecodingConfig,
        max_episodes: int,
        for_evaluator: Optional[bool] = False,
    ) -> Tuple[List[Episode], Dict]:
        """Generates samples using the inference engine."""
        torch.cuda.empty_cache()
        stats_tracker = self._init_stats_tracker()
        collected_episodes = []
        states = vector_env.reset()

        with tqdm(total=max_episodes, desc=f'Generating episodes', unit='episode') as pbar:
            while len(collected_episodes) < max_episodes:
                active_env_indices = [i for i, state in enumerate(states) if state is not None]
                active_states = [states[i] for i in active_env_indices]

                actions = [None] * vector_env.num_envs  # Generate actions only for active environments
                if active_states:
                    active_actions = self.act(active_states, decoding)
                    for i, action in zip(active_env_indices, active_actions):
                        actions[i] = action

                new_states = vector_env.step(actions)

                for i in range(vector_env.num_envs):  # Process completed episodes and reset
                    if states[i] is not None and vector_env.is_done(i):
                        episode = vector_env.get_episode(i)
                        processed_episodes = self._process_batch_episodes([episode], stats_tracker, for_evaluator)
                        if processed_episodes:
                            collected_episodes.extend(processed_episodes)
                            if len(collected_episodes) % 10 == 0:
                                pbar.update(10)
                        new_states[i] = vector_env.reset_one(i)

                states = new_states
                if self.tracker and len(collected_episodes) and len(collected_episodes) % 100 == 0:
                    Thread(target=self.tracker.flush).start()

        iter_stats = self._compute_iteration_stats(
            stats_tracker, collected_episodes, vector_env.max_reward, pbar.format_dict.get('elapsed', 0)
        )
        self._log_iteration_stats(iter_stats, for_evaluator)
        return collected_episodes, iter_stats

    def _init_stats_tracker(self) -> Dict:
        """Initialize statistics tracking dictionary."""
        return {'total': 0, 'correct': 0, 'bad': 0, 'skipped': 0}

    def _process_batch_episodes(self, batch_episodes: List[Episode], stats_tracker: Dict, for_evaluator: bool) -> List[Episode]:
        """Process and filter batch episodes."""
        current_batch_size = len(batch_episodes)
        stats_tracker['total'] += current_batch_size

        # if not for_evaluator:
        #     filtered_episodes = [sample for sample in batch_episodes if self._is_valid_sample(sample)]  # Filter bad samples
        #     bad_count = current_batch_size - len(filtered_episodes)
        #     stats_tracker['skipped'] += bad_count
        #     batch_episodes = filtered_episodes

        processed_episodes = []
        for episode in batch_episodes:
            processed_episodes.append(episode)
            self._log_episode(episode, for_evaluator)
        return processed_episodes

    def _is_valid_sample(self, sample: Episode) -> bool:
        """Check if a sample meets quality criteria."""
        if sample.count_completion_tokens() < 100:
            self.logger.debug('Skip too short episode')
            return False
        if any([True for t in sample.transitions if len(t.action.text) < 50]):
            self.logger.debug('Skip too short transition')
            return False
        return True

    def _log_episode(self, episode: Episode, for_evaluator: bool) -> None:
        """Log episode information if tracker is available."""
        if self.tracker:
            Thread(target=self.tracker.log_actor_step, args=(episode, for_evaluator)).start()

    def _log_iteration_stats(self, iter_stats: Dict, for_evaluator: bool) -> None:
        """Log iteration results and update tracker."""
        self.logger.info(f"Actor stats: {iter_stats}")
        if self.tracker:
            self.tracker.log_actor_iteration_stats(iter_stats, for_evaluator)

    def _compute_iteration_stats(
        self, stats_tracker: Dict, collected_episodes: List[Episode], max_reward: float, elapsed_time: float
    ) -> Dict:
        """Compute comprehensive iteration statistics."""

        # TODO convert to tensor and gather from all ranks
        episode_prompt_tokens = np.array([ep.count_prompt_tokens() for ep in collected_episodes])
        episode_completion_tokens = np.array([ep.count_completion_tokens() for ep in collected_episodes])
        episode_total_tokens = np.array([ep.count_total_tokens() for ep in collected_episodes])
        correct_samples = sum(1 for ep in collected_episodes if ep.graded_reward == max_reward)

        base_stats = {
            'elapsed/total_time': round(elapsed_time, 4),
            'elapsed/step_time': round(elapsed_time / max(stats_tracker['total'], 1), 4),
            'elapsed/total_episodes': stats_tracker['total'],
            'accuracy': correct_samples / stats_tracker['total'] if stats_tracker['total'] > 0 else 0.0,
            'stats/correct_ratio': correct_samples / len(collected_episodes) if collected_episodes else 0.0,
            'stats/skipped_ratio': stats_tracker['skipped'] / stats_tracker['total'] if stats_tracker['total'] > 0 else 0.0,
            'episode/completion_tokens': (
                np.mean(episode_completion_tokens).item() if episode_completion_tokens.size > 0 else 0.0
            ),
            'usage/total_tokens': np.sum(episode_total_tokens).item() if episode_total_tokens.size > 0 else 0.0,
            'usage/prompt_tokens': np.sum(episode_prompt_tokens).item() if episode_prompt_tokens.size > 0 else 0.0,
            'usage/completion_tokens': np.sum(episode_completion_tokens).item() if episode_completion_tokens.size > 0 else 0.0,
        }

        step_stats = self._compute_step_level_stats(collected_episodes)
        return {**base_stats, **step_stats}

    @staticmethod
    def _compute_step_level_stats(collected_episodes: List[Episode]) -> Dict:
        """Compute detailed statistics for each step."""
        steps_data = {}
        for episode in collected_episodes:
            for t in episode.transitions:
                step_name = t.state.state_id
                if step_name not in steps_data:
                    steps_data[step_name] = {'lengths': [], 'rewards': []}
                steps_data[step_name]['rewards'].append(t.reward)
                try:
                    steps_data[step_name]['lengths'].append(t.action.usage.completion_tokens)
                except Exception:  # Broad exception to catch cases where usage might be None
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
