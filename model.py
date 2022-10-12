import torch
from torch import nn

class SnakeBrain(torch.nn.Module):
    def __init__(self, input_size: int, output: int, frames: int):
        print("Initialized model")
        super(SnakeBrain, self).__init__()

        #Part 1
        self.part1 = nn.Sequential(
            nn.Conv2d(3*frames, 16, 2, padding = (1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 2),
            nn.BatchNorm2d(16),
            nn.ReLU()
        )

        #Part 2
        self.part2 = nn.Sequential(
        nn.Conv2d(3*frames, 16, 3, padding = (2, 2)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        self.linear_input = nn.Sequential(
            nn.Flatten(1),
            nn.Linear(49*3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
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
            nn.Linear(128, output),
        )      
        
    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()
        x = torch.concat((torch.flatten(self.part1(x), 1), torch.flatten(self.part2(x), 1)), axis = 1)
        #x = self.linear_input(x)
        #x = torch.flatten(x, 1)

        return self.linear(x)
"""
class SnakeBrain(torch.nn.Module):
    def __init__(self, input_size: int, output: int, frames: int):
        super(SnakeBrain, self).__init__()
        self.appel = SnakeModule(frames)
        self.head = SnakeModule(frames)
        self.body = SnakeModule(frames)
        self.combiner = torch.nn.Conv2d(4*3, 8, 3)

        conv_output = ((input_size-6)**2)*8
        self.fc1 = torch.nn.Linear(conv_output, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, output)

        self.relu = torch.nn.ReLU()        
        
    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()

        x = torch.concat((self.appel(x[:, 0:1]), self.body(x[:, 1:2]), self.head(x[:, 2:3])), axis=1)
        x = self.combiner(x)
        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x  


class SnakeModule(torch.nn.Module):
    def __init__(self, frames: int):
        super(SnakeModule, self).__init__()
        self.conv1 = torch.nn.Conv2d(frames, 4, 2)
        self.conv2 = torch.nn.Conv2d(4, 8, 2)
        self.conv3 = torch.nn.Conv2d(8, 4, 3)
        self.relu = torch.nn.ReLU()


    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        return x"""