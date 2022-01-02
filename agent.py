from typing import List
import torch
from memory_bank import Memory, MemoryBank
from game_map import Game
import random as rn
from new_trainer import DQTrainer

class DQAgent:
    def __init__(self, max_episodes: int, load_path: str = None, bank_size: int = 100000) -> None:
        self.bank: MemoryBank = MemoryBank(bank_size)
        self.testing = False
        self.episodes = 0
        if load_path is not None:
            model = torch.load(load_path)
            self.trainer: DQTrainer = DQTrainer(model = model)
        else:
            self.trainer: DQTrainer = DQTrainer()

        self.epsilon = 1/(max_episodes*1.1)
        self.previous_memory = None

    def _exploration_rate(self):
        return (1-(self.epsilon*self.episodes))

    def get_move(self, game: Game):
        if self.testing:
            prediction = self._predict(game)
            return prediction
        prediction = self._predict(game)
        if rn.random() < self._exploration_rate():
            prediction = rn.randint(0, 3)
            return prediction

    def make_memory(self, game: Game, move: int, illegal: bool) -> None:
        memory = Memory()
        memory.state = self._get_features(game.game_map)
        memory.action = move

        if self.previous_memory is not None:
            self.previous_memory.next_state = memory.state
            self.previous_memory.reward = self._get_reward(game, illegal)
            self.previous_memory.done = game.dead

            self.bank.addMemory(self.previous_memory)

        self.previous_memory = memory if not game.dead else None

                # if game.dead and not self.testing:
        #     self.episodes += 1
        #     self.previous_memory.done = True
        #     #Do this to make sure that during training we won't get None errors
        #     #And since we null out all done's anyway, the values here don't matter
        #     self.previous_memory.next_state = self._get_features(game.game_map)
        #     self.previous_memory.reward = self._get_reward(game)
        #     self.bank.addMemory(self.previous_memory)
        #     self.previous_memory = None
        #     return

        # if self.testing:
        #     prediction = self._predict(game)
        #     return prediction

        # memory = Memory()
        # prediction = self._predict(game)
        # if rn.random() < self._exploration_rate():
        #             prediction = rn.randint(0, 3)

        # memory.state = self._get_features(game.game_map)
        # memory.action = prediction
        

        # if self.previous_memory is not None and not self.previous_memory.done:
        #     self.previous_memory.next_state = memory.state
        #     self.previous_memory.reward = self._get_reward(game)
        #     self.bank.addMemory(self.previous_memory)

        # # if self.previous_memory is not None: self.bank.addMemory(self.previous_memory)
        # self.previous_memory = memory

        # return memory.action

    def train(self):
        self._train()

    def _get_features(self, game_map: List[List[int]]):
        features = torch.zeros((3, len(game_map), len(game_map)))
        for i in range(len(game_map)):
            for j in range(len(game_map)):
                tile = game_map[i][j]
                if tile != 0:
                    features[tile-1, i, j] = 1
        return features

    def _get_reward(self, game: Game, illegal: bool) -> float:
        if game.ate_last_turn:
            return 1.0
        if game.dead:
            return -1.0
        if illegal:
            return -1
        if game.distToApple() < game.previousAppleDistance:
            return 0.2
        if game.distToApple() >= game.previousAppleDistance:
            return -0.2


    def _predict(self, game: Game):
        self.trainer.model.eval()
        return torch.argmax(self.trainer.model(self._get_features(game.game_map)))

    def _train(self):
        self.trainer.model.train()
        self.trainer.train(self.bank)


        