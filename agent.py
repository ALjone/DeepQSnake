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

        self.trainer: DQTrainer = DQTrainer(hyperparams)
        self.train_times: int = hyperparams.train_times

        self.previous_state: torch.Tensor = None


    def get_move(self, state: np.ndarray, valid_moves: np.ndarray) -> int:

        return self._predict(state, valid_moves)

    def make_memory(self, action: int, state: np.ndarray, next_state: np.ndarray, reward: float, done: bool) -> None:
        #state = torch.from_numpy(state)
        #next_state = torch.from_numpy(next_state)
        #print(action.shape)
        self.bank.add((self.__get_features(state), torch.tensor(action), torch.tensor(reward), self.__get_features(next_state) if not done else torch.zeros(self.__get_features(next_state).shape), int(done)))

        self.previous_state = state if not done else None

    def train(self):
        """Call this after an episode is finished."""
        for _ in range(self.train_times):
            idxs, error = self.trainer.train(self.bank)
            self.bank.update_priorities(idxs, error)

    def __get_features(self, state: torch.Tensor):
        return state


    def _predict(self, state: torch.Tensor, valid_moves: torch.Tensor):
        return self.trainer.predict(self.__get_features(state), valid_moves)

    def _get_random(self, valid_moves: torch.Tensor):
        return self.trainer.random(valid_moves)