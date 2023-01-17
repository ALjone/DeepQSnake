import numpy as np
import torch
from hyperparams import Hyperparams
from network import DQTrainer
from prioritized_replay_memory import PrioritizedReplayBuffer
from tqdm import tqdm


class DQAgent:
    def __init__(self, hyperparams: Hyperparams) -> None:
        
        self.action_space = hyperparams.action_space
        self.bank: PrioritizedReplayBuffer = PrioritizedReplayBuffer((3*hyperparams.frame_stack, hyperparams.size, hyperparams.size), 1, hyperparams.replay_size, hyperparams.device, beta =hyperparams.beta, alpha = hyperparams.alpha)
        self.device = hyperparams.device
        self.testing: bool = False

        self.trainer: DQTrainer = DQTrainer(hyperparams)
        self.trained_times = 0
        self.bsz = hyperparams.batch_size
        self.memories_per_memory = hyperparams.workers

    def get_move(self, state: np.ndarray, valid_moves: np.ndarray) -> int:
        state = torch.tensor(state) if type(state) == np.ndarray else state
        valid_moves = torch.tensor(valid_moves) if type(valid_moves) == np.ndarray or type(valid_moves) == list else valid_moves

        return self._predict(state, valid_moves)

    def make_memory(self, memories: list):
        action, state, next_state, reward, done = memories
        state = torch.Tensor(state).flatten(1, 2)
        next_state = torch.Tensor(next_state).flatten(1, 2)
        self._make_memory(action, state, next_state, reward, done)
        #for action, state, next_state, reward, done in zip(*memories):
        #    self._make_memory(action, torch.Tensor(state).flatten(0, 1), torch.Tensor(next_state).flatten(0, 1), reward, done)

    def _make_memory(self, action: int, state: np.ndarray, next_state: np.ndarray, reward: float, done: bool) -> None:
        #state = torch.tensor(state) if type(state) == np.ndarray else state
        #next_state = torch.tensor(next_state) if type(next_state) == np.ndarray else next_state

        self.bank.add((state, action.unsqueeze(1), reward, next_state, done), num = self.memories_per_memory)

    def train(self, train_times, use_tqdm = False):
        """Call this after an episode is finished."""
        self.trained_times += train_times*self.bsz
        for _ in tqdm(range(train_times), leave=False, desc="Training", disable=~use_tqdm):
            idxs, error = self.trainer.train(self.bank)
            self.bank.update_priorities(idxs, error)


    def _predict(self, state: torch.Tensor, valid_moves: torch.Tensor):
        return self.trainer.predict(state, valid_moves)

    def _get_random(self, valid_moves: torch.Tensor):
        valid_moves = torch.tensor(valid_moves) if type(valid_moves) == np.ndarray else valid_moves
        return self.trainer.random(valid_moves)