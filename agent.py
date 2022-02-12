from typing import List
import torch
from hyperparams import Hyperparams
import random as rn
from network_trainer import DQTrainer
from ReplayMemory import ReplayMemory, Memory

class DQAgent:
    def __init__(self, hyperparams: Hyperparams) -> None:
        
        self.action_space = hyperparams.action_space
        self.bank: ReplayMemory = ReplayMemory(hyperparams.replay_size, hyperparams.apple_reward, hyperparams.death_reward)
        self.testing = False
        self.episodes = 0
        self.trainer: DQTrainer = DQTrainer(hyperparams)
        self.hyperparams = hyperparams
        self.game = hyperparams.game
        
        self.top_k = hyperparams.top_k
        self.epsilon = hyperparams.first_epsilon
        #TODO rename?
        self._exploration_rate_start = hyperparams.exploration_rate_start
        self._exploration_rate_end = hyperparams.exploration_rate_end

        self.previous_memory = None
        self.previous_state = None

    def _exploration_rate(self):
        return max(self._exploration_rate_start, self._exploration_rate_end)

    def get_move(self):
        if self.testing:
            prediction = self._predict()
            return prediction

        prediction = self._predict()
        if rn.random() < self._exploration_rate():
            #TODO Random sampling that can go into a wall, but not tail, at least to start with
            #TODO and then later not in a wall either.
            prediction = self._get_random(self.top_k)

        return prediction

    def make_memory(self, move: int) -> None:
        #TODO fix to not use memory?
        #TODO Rename to end of turn?
        memory = Memory()
        memory.state = self.__get_features()
        memory.action = move

        if self.previous_memory is not None:

            self.bank.push(self.previous_memory.state, torch.tensor([self.previous_memory.action]),
            memory.state if not self.game.final_state else None, torch.tensor(self.game.get_reward()))

        self.previous_memory = memory if not self.game.final_state else None
        self.previous_state = self.game.get_map() if not self.game.final_state else None

    def game_is_done(self):
        """Call this after an episode is finished."""
        self._exploration_rate_start -= self.epsilon
        if self._exploration_rate_start < self.hyperparams.epsilon_cutoff:
            self.epsilon = self.hyperparams.second_epsilon
        if self.episodes > self.hyperparams.lower_limit_1:
            self.top_k = self.hyperparams.top_k - 1
        if self.episodes > self.hyperparams.lower_limit_2:
            self.top_k = self.hyperparams.top_k - 2
        self.trainer.model.train()
        self.trainer.train(self.bank)

    def __get_features(self):
        if self.hyperparams.frame_stacks == 2:
            return torch.cat((self.game.get_map(), self.previous_state if self.previous_state is not None else self.game.get_map()), dim = 0)
        return self.game.get_map()


    def _predict(self):
        return self.trainer.predict(self.__get_features())

    def _get_random(self, top_x):
        return self.trainer.random(self.__get_features(), top_x)