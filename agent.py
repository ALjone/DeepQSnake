import numpy as np
import torch
from hyperparams import Hyperparams
import random as rn
from network_trainer import DQTrainer
from ReplayMemory import ReplayMemory, Memory

class DQAgent:
    def __init__(self, hyperparams: Hyperparams) -> None:
        
        self.action_space = hyperparams.action_space
        self.bank: ReplayMemory = ReplayMemory(hyperparams.replay_size, hyperparams.apple_reward, hyperparams.death_reward)
        self.testing: bool = False
        self.episodes: int = 0
        self.trainer: DQTrainer = DQTrainer(hyperparams)
        self.hyperparams: Hyperparams = hyperparams
        
        self.top_k: int = hyperparams.top_k
        self.epsilon: float = hyperparams.first_epsilon

        self._exploration_rate_curr: float = hyperparams.exploration_rate_start
        self._exploration_rate_end: float = hyperparams.exploration_rate_end

        self.previous_memory: bool = None
        self.previous_state: torch.Tensor = None

    def _exploration_rate(self) -> float:
        """Returns the exploration rate at this point in the training process"""
        return max(self._exploration_rate_curr, self._exploration_rate_end)

    def get_move(self, state: np.ndarray) -> int:
        state = torch.from_numpy(state)
        if self.testing:
            prediction = self._predict(state)
            return prediction

        prediction = self._predict(state)
        if rn.random() < self._exploration_rate():
            #TODO Random sampling that can go into a wall, but not tail, at least to start with
            #TODO and then later not in a wall either.
            prediction = self._get_random(self.top_k, state)

        return prediction

    def make_memory(self, action: int, state: np.ndarray, next_state: np.ndarray, reward: float, done: bool) -> None:
        #TODO fix to not use memory?
        #TODO Rename to end of turn?
        state = torch.from_numpy(state)
        next_state = torch.from_numpy(next_state)
        """memory = Memory()
        memory.state = self.__get_features(state)
        memory.action = action
        memory.reward = reward
        memory.next_state = next_state"""
        #TODO currently doesn't work with stack
        self.bank.push(self.__get_features(state), torch.tensor([action]), self.__get_features(next_state) if not done else None, torch.tensor(reward))
        """
        if self.previous_memory is not None:

            self.bank.push(self.previous_memory.state, torch.tensor([self.previous_memory.action]),
            memory.state if not done else None, torch.tensor(reward))

        self.previous_memory = memory if not done else None"""
        self.previous_state = state if not done else None

    def game_is_done(self):
        """Call this after an episode is finished."""
        self._exploration_rate_curr -= self.epsilon
        if self._exploration_rate_curr < self.hyperparams.epsilon_cutoff:
            self.epsilon = self.hyperparams.second_epsilon
        if self.episodes > self.hyperparams.lower_limit_1:
            self.top_k = self.hyperparams.top_k - 1
        if self.episodes > self.hyperparams.lower_limit_2:
            self.top_k = self.hyperparams.top_k - 2
        self.trainer.train(self.bank)

    def __get_features(self, state: torch.Tensor):
        if self.hyperparams.frame_stacks == 2:
            return torch.cat((state, self.previous_state if self.previous_state is not None else state), dim = 0)
        return state


    def _predict(self, state: torch.Tensor):
        return self.trainer.predict(self.__get_features(state))

    def _get_random(self, top_x: int, state: torch.Tensor):
        return self.trainer.random(self.__get_features(state), top_x)