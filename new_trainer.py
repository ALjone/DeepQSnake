from typing import List
from memory_bank import Memory, MemoryBank
import torch
from copy import deepcopy
from model import SnakeBrain
import numpy as np

class DQTrainer:
    def __init__(self) -> None:
        self.model: SnakeBrain = SnakeBrain(4)
        self.gamma: float = 0.4
        self.optim: torch.optim.Adam = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.train_steps: int = 0
        self.future_model: SnakeBrain = deepcopy(self.model)
        self.loss: torch.nn.MSELoss = torch.nn.MSELoss()


    def train(self, bank: MemoryBank, steps = None, print_ = False):
        #TODO Figure out these values
        steps = 500
        self.model.train()
        losses = []
        for _ in range(steps):
            self.train_steps += 1
            if self.train_steps % 20 == 0:
                self.future_model = deepcopy(self.model)


            loss_ = self._train_step(bank.getSamples(1)[0])
            losses.append(loss_)

        if print_:
           print("The loss is {0} for {1} trainings".format(sum(losses)/len(losses), steps))

    def _train_step(self, memory: Memory):
        self.model.zero_grad()
        self.future_model.zero_grad()
        self.optim.zero_grad()

        predictions = self.model(memory.state, memory.life)

        predictions = torch.mul(predictions, self.one_hot(memory.action))

        targets = torch.mul(self.one_hot(memory.action), self._target(memory.next_state, memory.reward, memory.life, memory.done).unsqueeze(1))

        output = self.loss(targets, predictions)

        output.backward()
        self.optim.step()

        return output

    def one_hot(self, a):
        return torch.from_numpy(np.eye(4)[a])

    def _target(self, next_states, rewards, life, done):
        with torch.no_grad():
            #Flip done because false = 0, and we want to remove it where it is 1
            tensor = torch.mul(torch.max(self.future_model(next_states, life)), ~done)
            futures = torch.add(rewards, torch.mul(tensor, self.gamma))
        return futures
