"""Custom MDP environment."""

import logging
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
from datasets import Dataset

from rl4llm.graders import math_problem_grader
from rl4llm.types import ChatTurn, EnvAction, EnvState, Episode, StateNode
from rl4llm.utils import is_texts_similar

logger = logging.getLogger()


class StateGraph:
    """Structure to manage state transitions."""

    TERMINAL_STATE_ID = "__terminal__"

    def __init__(self):
        self.states: Dict[str, StateNode] = {}
        self._create_terminal_state()

    def _create_terminal_state(self):
        """Create the special terminal state"""
        terminal_state = StateNode(
            state_id=self.TERMINAL_STATE_ID,
            user_prompt="Episode complete.",
            transitions=[],
        )
        self.states[self.TERMINAL_STATE_ID] = terminal_state

    @classmethod
    def from_metadata(cls, metadata: List[Dict]) -> "StateGraph":
        """Create a state graph from metadata"""
        graph = cls()

        # Add states
        for data in metadata:
            graph._add_state(
                state_id=data["state_id"],
                user_prompt=data["user_prompt"],
                system_prompt=data.get("system_prompt", None),
                should_grade=data.get("should_grade", False),
                should_augment=data.get("should_augment", False),
                is_start=data.get("is_start", False),
            )

        # Add transitions
        for data in metadata:
            transitions = data.get("transitions", [])
            if not transitions:
                # If no transitions specified, automatically transition to terminal state
                transitions = [{"state_id": cls.TERMINAL_STATE_ID, "prob": 1.0}]
            for t in transitions:
                # If transition is to terminal marker, replace with actual terminal state ID
                to_state = cls.TERMINAL_STATE_ID if t.get("terminal", False) or t["state_id"] == "terminal" else t["state_id"]
                graph._add_transition(data["state_id"], to_state, t["prob"])

        graph._validate()
        return graph

    def get_state(self, state_id: str) -> Optional[StateNode]:
        """Get a state by ID"""
        return self.states.get(state_id)

    def get_start_state(self) -> StateNode:
        """Get the start state"""
        return [state for state in self.states.values() if state.is_start][0]

    def get_terminal_state(self) -> StateNode:
        """Get the terminal state"""
        return self.states[self.TERMINAL_STATE_ID]

    def _add_state(
        self,
        state_id: str,
        user_prompt: str,
        system_prompt: str,
        should_grade: bool = False,
        should_augment: bool = False,
        is_start: bool = False,
    ) -> StateNode:
        """Add a new state to the graph"""
        state = StateNode(
            state_id=state_id,
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            should_grade=should_grade,
            should_augment=should_augment,
            is_start=is_start,
        )
        self.states[state_id] = state
        return state

    def _add_transition(self, from_id: str, to_id: str, probability: float) -> None:
        """Add a transition between states"""
        from_state = self.states.get(from_id)
        to_state = self.states.get(to_id)
        if from_state and to_state:
            from_state.add_transition(to_state, probability)

    def _validate(self) -> bool:
        """Validate graph structure"""
        # Must have exactly one start state
        if len([state for state in self.states.values() if state.is_start]) != 1:
            raise ValueError("Graph must have exactly one start state")

        # Validate transition probabilities and ensure all paths lead to terminal
        for state in self.states.values():
            if state.state_id != self.TERMINAL_STATE_ID:
                total_prob = sum(prob for _, prob in state.transitions)
                if abs(total_prob - 1.0) > 1e-8:
                    raise ValueError(f"Transitions for state {state.state_id} sum to {total_prob}, not 1.0")

        # Verify all states can reach terminal state
        self._verify_terminal_reachable()
        return True

    def _verify_terminal_reachable(self):
        """Verify all states can reach the terminal state"""

        def can_reach_terminal(state: StateNode, visited=None):
            if visited is None:
                visited = set()

            if state.state_id in visited:
                return False

            if state.state_id == self.TERMINAL_STATE_ID:
                return True

            visited.add(state.state_id)
            return any(can_reach_terminal(next_state, visited.copy()) for next_state, _ in state.transitions)

        for state in self.states.values():
            if not can_reach_terminal(state):
                raise ValueError(f"State {state.state_id} cannot reach terminal state")


class MDPEnv:
    """Custom MDP environment with improved state management."""

    def __init__(
        self,
        dataset: Dataset,
        state_config: List[Dict],
        stop_tokens: List[str] = [],
        min_reward: float = 0.0,
        max_reward: float = 1.0,
        mid_step_score: float = 0.5,
        seed: int = 42,
    ):
        assert min_reward < max_reward, "min_reward must be less than max_reward"
        assert 0 <= mid_step_score < max_reward, "invalid mid_step_score"
        assert state_config, "state_config cannot be empty"
        assert len(stop_tokens) >= 1 and all([isinstance(d, str) and " " not in d for d in stop_tokens]), "invalid stop_tokens"

        # Initialize random seed
        self.seed = seed
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)

        # Initialize dataset
        self.dataset = dataset.shuffle(seed=self.seed)
        self.dataset_iterator = iter(self.dataset)

        # Store configuration
        self.stop_tokens = stop_tokens
        self.max_reward = max_reward
        self.min_reward = min_reward
        self.mid_step_score = mid_step_score

        # Initialize state management
        self.state_graph = StateGraph.from_metadata(state_config)
        self.initial_state = self.state_graph.get_start_state()
        self.terminal_state = self.state_graph.get_terminal_state()

        # Initialize episode state
        self.curr_state = None
        self.curr_episode = None
        self.done = False
        self.step_t = 0
        self.num_episodes = 0

    def reset(self) -> EnvState:
        """Reset the environment and start a new episode."""
        # Get next sample
        try:
            sample = next(self.dataset_iterator)
        except StopIteration:
            self.dataset = self.dataset.shuffle(seed=random.randint(0, 1000))
            self.dataset_iterator = iter(self.dataset)
            sample = next(self.dataset_iterator)

        # Reset episode state
        self.step_t = 0
        self.curr_state = self.initial_state
        self.done = False

        # Initialize new episode
        self.curr_episode = Episode(
            question=sample["question"],
            ground_truth=sample["ground_truth"],
            task_type=str(sample["task_type"]).strip().upper(),
            transitions=[],
        )

        return self._build_external_state()

    def step(self, action: EnvAction) -> EnvState:
        """Take a step in the environment."""
        if self.curr_state is None or self.is_done():
            raise RuntimeError("Call reset before continuing or episode has terminated.")
        if not action or not isinstance(action, EnvAction):
            raise ValueError("Invalid action.")
        if not action.text or not action.text.strip():
            raise ValueError("Invalid action: Action must contain text.")

        # Pre-process action text
        action = self._clean_action(action)

        # Update state
        self._state_transition(action)

        if self.is_done():
            self.num_episodes += 1
            return None

        return self._build_external_state()

    def _state_transition(self, action: EnvAction):
        """Update environment state after an action."""
        reward = 0.0

        short_answer = None
        # Handle grading if needed
        if self.curr_state.should_grade:
            reward, short_answer = self._compute_graded_reward(action.text)

        next_state = self._get_next_state()
        is_done = next_state == self.terminal_state or self.done

        self.curr_episode.record_transition(
            state=self.curr_state,
            action=action,
            reward=reward,
            done=is_done,
        )

        if is_done:
            self._handle_termination(reward=reward, short_answer=short_answer)
            return

        self.curr_state = next_state
        self.step_t += 1

    def _build_external_state(self) -> List[ChatTurn]:
        """Build external state based on current state and action."""
        external_state: List[ChatTurn] = []

        for i, t in enumerate(self.curr_episode.transitions):
            # for user turn
            user_content = self._format_user_content(t.state.user_prompt)
            external_state.append(ChatTurn(role="user", content=user_content))

            # for assistant turn
            assistant_content = t.action.text
            external_state.append(ChatTurn(role="user", content=assistant_content))

        # add last user turn which is current state
        user_content = self._format_user_content(self.curr_state.user_prompt)
        external_state.append(ChatTurn(role="user", content=user_content))

        # always use step-specific system-level prompt if available, otherwise reset it
        if self.curr_state.system_prompt:
            curr_sys_turn = ChatTurn(role="system", content=self.curr_state.system_prompt)
            if external_state and external_state[0].role == "system":
                external_state[0] = curr_sys_turn
            else:
                external_state.insert(0, curr_sys_turn)
        elif external_state and external_state[0].role == "system":
            external_state.pop(0)

        return external_state

    def _clean_action(self, action: EnvAction) -> EnvAction:
        action_text = action.text
        for tk in self.stop_tokens:
            action_text = action_text.replace(tk, "")
        action.text = action_text.strip()
        return action

    def _format_user_content(self, user_prompt: str) -> str:
        """Format user chat content with question."""
        if self.step_t == 0:
            # add original question for the initial step
            question = self.curr_episode.question.strip()
            try:
                return user_prompt.format(question=question).strip()
            except Exception as _e:
                return f"{user_prompt.strip()}\n\n{question}"
        else:
            return user_prompt.strip()

    def _compute_graded_reward(self, full_answer: str) -> Tuple[float, str]:
        if not full_answer:
            logger.warning("Received an invalid full answer.")
            return self.min_reward, "[INVALID_ANSWER]"

        task_type = self.curr_episode.task_type
        if task_type in ["GSM", "MATH"]:
            graded_reward, short_answer = math_problem_grader(
                full_answer=full_answer,
                ground_truth=self.curr_episode.ground_truth,
            )
            logger.debug(f"Graded reward: {graded_reward}")

            # scale reward if required
            if self.max_reward > 1 and graded_reward == 1:
                graded_reward = self.max_reward
            elif self.min_reward < 0 and graded_reward == 0:
                graded_reward = self.min_reward
            return graded_reward, short_answer
        else:
            raise RuntimeError(f"Unsupported task type {task_type}")

    def _get_next_state(self) -> StateNode:
        """Get next state based on transition probabilities"""
        if self.done:
            return None

        if not self.curr_state.transitions:
            return None

        possible_states = [state for state, prob in self.curr_state.transitions]
        probabilities = [prob for state, prob in self.curr_state.transitions]

        # random.choices returns a list, even if k=1
        next_states = random.choices(possible_states, weights=probabilities, k=1)
        return next_states[0]

    def _handle_termination(self, reward: float, short_answer: str):
        """Handle episode terminal"""
        self.curr_episode.short_answer = short_answer
        self.curr_episode.graded_reward = reward
        for t in self.curr_episode.transitions:
            # Assign middle-step reward based on graded_reward condition
            if not t.is_done and t.reward == 0:
                t.reward = self.min_reward if reward == self.min_reward else self.mid_step_score

        self.done = True
        self.curr_state = None

    def _is_similar_to_previous(self, curr_text: str, ratio: float = 0.75) -> bool:
        if not self.curr_episode or len(self.curr_episode.transitions) == 0:
            return False
        last_text = self.curr_episode.transitions[-1].action.text
        if not curr_text or not last_text:
            return False

        return is_texts_similar(curr_text, last_text, ratio)

    def is_done(self) -> bool:
        return self.done

    def get_current_episode(self) -> Episode:
        if self.is_done() and self.curr_episode:
            return self.curr_episode
        raise RuntimeError("Episode is not done yet.")


if __name__ == "__main__":

    from rl4llm.data import load_gsm_dataset

    train_ds, test_ds = load_gsm_dataset()

    state_config = [
        {
            "state_id": "start",
            "system_prompt": "You are a very smart model",
            "user_prompt": "Help me to analyze this problem in details.\n\n{question}",
            "is_start": True,
            "transitions": [
                {"state_id": "followup", "prob": 0.5},
                {"terminal": True, "prob": 0.7},
            ],
        },
        {
            "state_id": "followup",
            "system_prompt": "Be testing system prompt???",
            "user_prompt": "What is 2+2?",
            "should_grade": True,
            "should_augment": True,
            "transitions": [{"terminal": True, "prob": 1.0}],
        },
    ]

    env = MDPEnv(
        train_ds,
        state_config=state_config,
        stop_tokens=["<|im_end|>"],
    )

    def selfplay_policy(state, t):
        import time

        return EnvAction(
            **{
                "text": f"Here is the random action at step {t} on {time.time()} ...<|im_end|>",
                "token_ids": [1, 3, 134, 343],
            }
        )

    for _ in range(10):
        state = env.reset()
        print(f"Step {env.step_t}: {state}\n##########")

        while env.is_done() is False:
            action = selfplay_policy(state, env.step_t)
            next_state = env.step(action)
            state = next_state
            # print(f"Step {env.step_t}: {state!r}\n\n")
            print(f"Step {env.step_t}: {state}\n##########")

        episode = env.get_current_episode()
        print(episode)
