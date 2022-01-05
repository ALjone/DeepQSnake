from typing import List
import torch
from hyperparams import Hyperparams
from memory_bank import Memory
from game import Game
import random as rn
from trainer import DQTrainer
from ReplayMemory import ReplayMemory

class DQAgent:
    def __init__(self, hyperparams: Hyperparams) -> None:
        
        self.action_space = hyperparams.action_space
        self.device = hyperparams.device
        self.bank: ReplayMemory = ReplayMemory(hyperparams.replay_size)
        self.testing = False
        self.episodes = 0
        self.trainer: DQTrainer = DQTrainer(hyperparams)


        self.epsilon = hyperparams.epsilon
        self._exploration_rate_start = hyperparams.exploration_rate_start
        self._exploration_rate_end = hyperparams.exploration_rate_end

        self.previous_memory = None

    def _exploration_rate(self):
        return max((self._exploration_rate_start-(self.epsilon*self.episodes)), self._exploration_rate_end)

    def get_move(self, game: Game):
        if self.testing:
            prediction = self._predict(game)
            return prediction

        prediction = self._predict(game)
        if rn.random() < self._exploration_rate():
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

    def train(self):
        #Assumes training only ones per game
        self.episodes += 1
        self.trainer.model.train()
        self.trainer.train(self.bank)

    def _predict(self, game: Game):
        self.trainer.model.eval()
        return torch.argmax(self.trainer.model(game.get_map().to(self.device)))