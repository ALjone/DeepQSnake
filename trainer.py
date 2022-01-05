from typing import List
from ReplayMemory import ReplayMemory, Transition
import torch
from hyperparams import Hyperparams
from model import SnakeBrain


class DQTrainer:
    def __init__(self, hyperparams: Hyperparams) -> None:
        self.model: SnakeBrain = SnakeBrain(hyperparams.game.mapsize, hyperparams.action_space) if hyperparams.load_path is None else torch.load(hyperparams.load_path)
        #Try high
        self.gamma: float = hyperparams.gamma
        self.optim: torch.optim.Adam = torch.optim.Adam(self.model.parameters(), lr=hyperparams.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.future_model: SnakeBrain = SnakeBrain(hyperparams.game.mapsize, hyperparams.action_space)
        self.future_model.load_state_dict(self.model.state_dict())

        self.model.to(self.device)
        self.future_model.to(self.device)

        self.loss: torch.nn.MSELoss = torch.nn.MSELoss()

        self.prime_update_rate: int = hyperparams.prime_update_rate
        self.episodes: int = 0
        self.batch_size = hyperparams.batch_size



    def train(self, bank: ReplayMemory, print_ = False) -> None:
        if len(bank) < self.batch_size:
            return
        self.model.train()
        losses = []
        self.episodes += 1
        if self.episodes % self.prime_update_rate == 0:
            #print("Copied!")
            self.future_model.load_state_dict(self.model.state_dict())

        samples = bank.sample(self.batch_size)
        loss_ = self._train_batch(samples)
        losses.append(loss_)

        if print_:
           print("Episode:", self.episodes, "The loss is {0} for batch size {1}".format(sum(losses)/len(losses), self.batch_size))


    def _train_batch(self, memories: List[Transition]):
        self.model.zero_grad()

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
        #TODO used to say rewards, which is same as setting gamma to 0
        return targets
