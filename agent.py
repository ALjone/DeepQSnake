import numpy as np
import torch
from hyperparams import Hyperparams
import random as rn
from network_trainer import DQTrainer
from prioritized_replay_memory import PrioritizedReplayBuffer

class DQAgent:
    def __init__(self, hyperparams: Hyperparams) -> None:
        
        self.action_space = hyperparams.action_space
        self.bank: PrioritizedReplayBuffer = PrioritizedReplayBuffer((3, hyperparams.size, hyperparams.size), 1, hyperparams.replay_size, hyperparams.device, beta =hyperparams.beta, alpha = hyperparams.alpha)
        self.testing: bool = False
        self.episodes: int = 0
        self.trainer: DQTrainer = DQTrainer(hyperparams)
        self.hyperparams: Hyperparams = hyperparams
        
        self.epsilon: float = hyperparams.epsilon

        self._exploration_rate_curr: float = hyperparams.exploration_rate_start
        self._exploration_rate_end: float = hyperparams.exploration_rate_end

        self.previous_state: torch.Tensor = None

    def _exploration_rate(self) -> float:
        """Returns the exploration rate at this point in the training process"""
        return max(self._exploration_rate_curr, self._exploration_rate_end)

    def get_move(self, state: np.ndarray, valid_moves: np.ndarray) -> int:
        if self.testing:
            prediction = self._predict(state, valid_moves)
            return prediction

        
        if rn.random() < self._exploration_rate():
            #TODO Random sampling that can go into a wall, but not tail, at least to start with
            #TODO and then later not in a wall either.
            return self._get_random(valid_moves)

        return self._predict(state, valid_moves)

    def make_memory(self, action: int, state: np.ndarray, next_state: np.ndarray, reward: float, done: bool) -> None:
        #state = torch.from_numpy(state)
        #next_state = torch.from_numpy(next_state)
        #print(action.shape)
        self.bank.add((self.__get_features(state), torch.tensor(action), torch.tensor(reward), self.__get_features(next_state) if not done else torch.zeros(self.__get_features(next_state).shape), int(done)))

        self.previous_state = state if not done else None

    def game_is_done(self):
        """Call this after an episode is finished."""
        self._exploration_rate_curr -= self.epsilon
        for i in range(self.hyperparams.train_times):
            idxs, error = self.trainer.train(self.bank)
            self.bank.update_priorities(idxs, error)

    def __get_features(self, state: torch.Tensor):
        return state


    def _predict(self, state: torch.Tensor, valid_moves: torch.Tensor):
        return self.trainer.predict(self.__get_features(state), valid_moves)

    def _get_random(self, valid_moves: torch.Tensor):
        return self.trainer.random(valid_moves)