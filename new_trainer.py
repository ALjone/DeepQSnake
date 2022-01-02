from typing import List
from memory_bank import Memory, MemoryBank
import torch
from model import SnakeBrain


class DQTrainer:
    def __init__(self, model: SnakeBrain = SnakeBrain(4)) -> None:
        self.model: SnakeBrain = model
        #Try high
        self.gamma: float = 0.95
        self.optim: torch.optim.Adam = torch.optim.Adam(self.model.parameters(), lr=0.001)


        self.future_model: SnakeBrain = SnakeBrain(4)
        self.future_model.load_state_dict(self.model.state_dict())

        self.loss: torch.nn.MSELoss = torch.nn.MSELoss()

        self.prime_update_rate: int = 10
        self.episodes: int = 0


    def train(self, bank: MemoryBank, steps = 100, print_ = False) -> None:
        self.model.train()
        losses = []
        self.episodes += 1
        if self.episodes % self.prime_update_rate == 0:
            self.future_model.load_state_dict(self.model.state_dict())

        for _ in range(steps):
            loss_ = self._train_step(bank.getSamples(1)[0])
            losses.append(loss_)

        if print_:
           print("The loss is {0} for {1} trainings".format(sum(losses)/len(losses), steps))

    def _train_step(self, memory: Memory):
        self.model.zero_grad()
        self.future_model.zero_grad()
        self.optim.zero_grad()

        predictions = self.model(memory.state)[0]
        predictions = predictions[memory.action]

        targets = self._target(memory.next_state, memory.reward, memory.done)

        output = self.loss(targets, predictions)

        output.backward()
        self.optim.step()

        return output

    def _target(self, next_states, reward, done):
        with torch.no_grad():
            #Flip done because false = 0, and we want to remove it where it is 1
            tensor = torch.max(self.future_model(next_states)) * ~done
            future = torch.add(reward, tensor * self.gamma)
        return future
