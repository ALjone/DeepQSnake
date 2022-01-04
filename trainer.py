from typing import List
from ReplayMemory import ReplayMemory, Transition
import torch
from model import SnakeBrain


class DQTrainer:
    def __init__(self, model: SnakeBrain = SnakeBrain(4)) -> None:
        self.model: SnakeBrain = model
        #Try high
        self.gamma: float = 0.999
        self.optim: torch.optim.Adam = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.future_model: SnakeBrain = SnakeBrain(4)
        self.future_model.load_state_dict(self.model.state_dict())

        self.model.to(self.device)
        self.future_model.to(self.device)

        self.loss: torch.nn.MSELoss = torch.nn.SmoothL1Loss()#MSELoss()

        self.prime_update_rate: int = 40
        self.episodes: int = 0



    def train(self, bank: ReplayMemory, steps = 512, print_ = False) -> None:
        if len(bank) < steps:
            return
        self.model.train()
        losses = []
        self.episodes += 1
        if self.episodes % self.prime_update_rate == 0:
            self.future_model.load_state_dict(self.model.state_dict())

        samples = bank.sample(steps)
        loss_ = self._train_batch(samples)
        #losses.append(loss_)

        #if print_:
        #   print("The loss is {0} for {1} trainings".format(sum(losses)/len(losses), steps))


    def _train_batch(self, memories: List[Transition]):

        batch: Transition = Transition(*zip(*memories))

        states = torch.stack(batch.state).to(self.device)
        actions = torch.stack(batch.action).to(self.device)

        done_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)

        next_states = torch.stack([s for s in batch.next_state
                                                if s is not None])
        
        rewards = torch.stack(batch.reward).to(self.device)

        predictions = self.model(states).gather(1, actions)


        targets = self._target(next_states, rewards, done_mask)

        #print(targets.unsqueeze(1))
        #print(predictions)
        #print(targets.unsqueeze(1)-predictions)
        #print("Diff:", torch.max(predictions).item()-torch.min(predictions).item())
        #print("Size of biggest:", torch.max(predictions).item())
        #input()

        loss = self.loss(predictions, targets.unsqueeze(1))
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return loss

    def _target(self, next_states, rewards, done_mask):
        with torch.no_grad():
            self.future_model.eval()
            #Flip done because false = 0, and we want to remove it where it is 1
            future_values = torch.zeros(done_mask.shape[0], device=self.device)
            future_values[done_mask] = self.future_model(next_states).max(1)[0].detach()
            targets = rewards + (future_values * self.gamma)
        return targets
