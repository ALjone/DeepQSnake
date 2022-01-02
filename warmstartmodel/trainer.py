from typing import List
import torch
from copy import deepcopy
from model import SnakeBrain

class DQTrainer:
    def __init__(self) -> None:
        self.model: SnakeBrain = SnakeBrain(4)
        self.optim: torch.optim.Adam = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.train_steps: int = 0
        self.loss: torch.nn.MSELoss = torch.nn.MSELoss()
        self.batch_size = 512


    def train(self, X, y, epochs = 50):
        train_dataset = torch.utils.data.TensorDataset(X,y)
        #Create the dataloader with the given hyperparameters
        train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=self.batch_size,
                                                  shuffle=True)
        for epoch in range(epochs):
            losses = []
            for i, data in enumerate(train_loader):
                images, labels = data
                self.optim.zero_grad()
                
                predictions = self.model(images)
                output = self.loss(labels, predictions)
                losses.append(output.item())

                output.backward()
                self.optim.step()

                if i % 10 == 0:
                    print("[{0}, {1}] The loss is {2}".format(epoch+1, i, sum(losses)/len(losses)))
        return output

        