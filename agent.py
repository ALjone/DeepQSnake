from typing import List
import torch
from hyperparams import Hyperparams
from game import Game
import random as rn
from network_trainer import DQTrainer
from ReplayMemory import ReplayMemory, Memory

class DQAgent:
    def __init__(self, hyperparams: Hyperparams) -> None:
        
        self.action_space = hyperparams.action_space
        self.device = hyperparams.device
        self.bank: ReplayMemory = ReplayMemory(hyperparams.replay_size)
        self.testing = False
        self.episodes = 0
        self.trainer: DQTrainer = DQTrainer(hyperparams)
        self.hyperparams = hyperparams

        self.epsilon = hyperparams.first_epsilon
        #TODO rename?
        self._exploration_rate_start = hyperparams.exploration_rate_start
        self._exploration_rate_end = hyperparams.exploration_rate_end

        self.previous_memory = None

    def _exploration_rate(self):
        return max(self._exploration_rate_start, self._exploration_rate_end)

    def get_move(self, game: Game):
        if self.testing:
            prediction = self._predict(game)
            return prediction

        prediction = self._predict(game)
        if rn.random() < self._exploration_rate():
            #TODO Random sampling that can go into a wall, but not tail, at least to start with
            #TODO and then later not in a wall either.
            prediction = rn.randint(0, self.action_space-1)

        return  prediction

    def make_memory(self, game: Game, move: int) -> None:
        #TODO fix to not use memory?
        memory = Memory()
        memory.state = game.get_map()
        memory.action = move

        if self.previous_memory is not None:

            self.bank.push(self.previous_memory.state, torch.tensor([self.previous_memory.action]),
            memory.state if not game.final_state else None, torch.tensor(game.get_reward()))

        self.previous_memory = memory if not game.final_state else None

    def game_is_done(self):
        """Call this after an episode is finished."""
        self._exploration_rate_start -= self.epsilon
        if self._exploration_rate_start < self.hyperparams.epsilon_cutoff:
            self.epsilon = self.hyperparams.second_epsilon
        self.trainer.model.train()
        self.trainer.train(self.bank)

    def _predict(self, game: Game):
        #TODO should call a method in trainer
        self.trainer.model.eval()
        return torch.argmax(self.trainer.model(game.get_map().to(self.device)))