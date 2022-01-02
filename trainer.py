from typing import List

from torch.cuda import memory
from memory_bank import Memory, MemoryBank
import torch
from model import SnakeBrain


class DQTrainer:
    def __init__(self, model: SnakeBrain = SnakeBrain(4)) -> None:
        self.model: SnakeBrain = model
        #Try high
        self.gamma: float = 0.95
        self.optim: torch.optim.Adam = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.future_model: SnakeBrain = SnakeBrain(4)
        self.future_model.load_state_dict(self.model.state_dict())

        self.model.to(self.device)
        self.future_model.to(self.device)

        self.loss: torch.nn.MSELoss = torch.nn.MSELoss()

        self.prime_update_rate: int = 10
        self.episodes: int = 0


    def train(self, bank: MemoryBank, steps = 512, print_ = False) -> None:
        self.model.train()
        losses = []
        self.episodes += 1
        if self.episodes % self.prime_update_rate == 0:
            self.future_model.load_state_dict(self.model.state_dict())

        # for _ in range(steps):
        #     loss_ = self._train_step(bank.getSamples(1)[0])
        #     losses.append(loss_)

        samples = bank.getSamples(steps)
        loss_ = self._train_step(samples)
        losses.append(loss_)

        if print_:
           print("The loss is {0} for {1} trainings".format(sum(losses)/len(losses), steps))

    def _train_step(self, memories: List[Memory]):
        self.model.zero_grad()
        self.future_model.zero_grad()
        self.optim.zero_grad()

        states = torch.cat([memory.state.unsqueeze(0) for memory in memories], dim=0)
        states = states.to(self.device)

        next_states = torch.cat([memory.next_state.unsqueeze(0) for memory in memories], dim=0)
        next_states = next_states.to(self.device)

        # actions = torch.tensor([memory.action for memory in memories])
        not_dones = torch.tensor([~memory.done for memory in memories]).to(self.device)
        rewards = torch.tensor([memory.reward for memory in memories]).to(self.device)

        predictions = self.model(states)
        predictions = torch.tensor([prediction[memory.action].item() for memory, prediction in zip(memories, predictions)], requires_grad=True).to(self.device)

        targets = self._target(next_states, rewards, not_dones)

        output = self.loss(targets, predictions)

        output.backward()
        self.optim.step()

        return output

    def _target(self, next_states, rewards, not_dones):
        with torch.no_grad():
            #Flip done because false = 0, and we want to remove it where it is 1
            tensor = torch.mul(torch.max(self.future_model(next_states)), not_dones)
            future = torch.add(rewards, tensor * self.gamma)
        return future
