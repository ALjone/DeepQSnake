from typing import List
from memory_bank import Memory, MemoryBank
import torch
from copy import deepcopy
from model import SnakeBrain
import numpy as np

class DQTrainer:
    def __init__(self, model: SnakeBrain = None) -> None:
        self.model: SnakeBrain = SnakeBrain(4)
        self.gamma: float = 0.7
        self.train_steps: int = 0
        self.loss: torch.nn.MSELoss = torch.nn.MSELoss()
        self.prime_update_rate: int = 200

        if model is not None:
            self.model.load_model(model.state_dict())

        self.optim: torch.optim.Adam = torch.optim.Adadelta(self.model.parameters(), lr=0.001)
        self.future_model: SnakeBrain = deepcopy(self.model)

    def train(self, bank: MemoryBank, steps = None, print_ = False):
        #TODO Figure out these values
        steps = 50
        samples = 400
        self.model.train()
        losses = []
        for _ in range(steps):
            self.train_steps += 1
            if self.train_steps % self.prime_update_rate == 0:
                self.future_model = deepcopy(self.model)


            loss_ = self._train_step(bank.getSamples(samples))
            losses.append(loss_)

        if print_:
           print("The loss is {0} for {1} trainings".format(sum(losses)/len(losses), steps*samples))

    def _train_step(self, memories: List[Memory]):
        #TODO Have now removed only grad for 1 of the outputs, is this correct or not???
        self.model.zero_grad()
        self.future_model.zero_grad()
        self.optim.zero_grad()

        features = torch.zeros((len(memories), 3, 10, 10))
        life = torch.zeros(len(memories))
        next_features = torch.zeros((len(memories), 3, 10, 10))
        rewards = torch.zeros(len(memories))
        agent_actions = torch.zeros((len(memories))).type(torch.LongTensor)
        done = torch.zeros(len(memories)).type(torch.BoolTensor)
       

        for i, memory in enumerate(memories):
            features[i] = memory.state
            life[i] = memory.life
            next_features[i] = memory.next_state
            rewards[i] = memory.reward
            agent_actions[i] = memory.action
            done[i] = memory.done
        
        #predictions = self.model(features, life)
        #predictions = torch.mul(predictions, self.one_hot(agent_actions))
        #targets = torch.mul(self.one_hot(agent_actions), self._target(next_features, rewards, life, done).unsqueeze(1))
        #output = self.loss(targets, predictions)
        predictions, _ = self.model(features, life).max(1)
        targets = self._target(next_features, rewards, life, done)
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
