from typing import List
import torch
from memory_bank import Memory, MemoryBank
from game import Game
import random as rn
from trainer import DQTrainer
from ReplayMemory import ReplayMemory

class DQAgent:
    def __init__(self, max_episodes: int, load_path: str = None, bank_size: int = 100000) -> None:
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bank: ReplayMemory = ReplayMemory(bank_size)
        self.testing = False
        self.episodes = 0
        if load_path is not None:   
            model = torch.load(load_path)
            self.trainer: DQTrainer = DQTrainer(model = model)
        else:
            self.trainer: DQTrainer = DQTrainer()

        #TODO should this be over 1?
        self.epsilon = 1/(max_episodes*0.9)
        self.previous_memory = None

    def _exploration_rate(self):
        return max((0.9-(self.epsilon*self.episodes)), 0.01)

    def get_move(self, game: Game):
        """True if prediction was made by the model, false if random"""
        if self.testing:
            prediction = self._predict(game)
            return (True, prediction)
        prediction = self._predict(game)
        if rn.random() < self._exploration_rate():
            prediction = rn.randint(0, 3)
            return (False, prediction)
        return (True, prediction)

    def make_memory(self, game: Game, move: int) -> None:
        memory = Memory()
        memory.state = game.get_map()
        memory.action = move

        if self.previous_memory is not None:

            self.bank.push(self.previous_memory.state, torch.tensor([self.previous_memory.action]),
            memory.state if not game.dead else None, torch.tensor(self._get_reward(game)))

        self.previous_memory = memory if not game.dead else None

    def train(self):
        #Assumes training only ones per game
        self.episodes += 1
        self.trainer.model.train()
        self.trainer.train(self.bank)


    def _get_reward(self, game: Game) -> float:
        if game.ate_last_turn:
            return 10.0
        if game.dead:
            return -10.0
        if game.distToApple() < game.previousAppleDistance:
            return 0.2
        if game.distToApple() >= game.previousAppleDistance:
            return -0.2


    def _predict(self, game: Game):
        self.trainer.model.eval()
        return torch.argmax(self.trainer.model(game.get_map().to(self.device)))