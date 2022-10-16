import torch
from torch import nn

class SnakeBrain(torch.nn.Module):
    def __init__(self, input_size: int, output: int):
        super(SnakeBrain, self).__init__()

        #Part 1
        self.part1 = nn.Sequential(
            nn.Conv2d(3, 16, 2, padding = (1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        #Part 2
        self.part2 = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding = (2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        conv_output = ((input_size-3)**2)*16*2

        self.linear = nn.Sequential(
            nn.Linear(conv_output, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),   
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
            )

        #Value of state
        self.V = nn.Sequential(
            nn.Linear(64, 1),
        )   
        #Impact of action
        self.A = nn.Sequential(
            nn.Linear(64, output),
        )   
        
    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()
        x = torch.concat((torch.flatten(self.part1(x), 1), torch.flatten(self.part2(x), 1)), axis = 1)
        x = self.linear(x)
        A = self.A(x)
        return self.V(x) + (A-A.mean(dim = 1, keepdim = True))