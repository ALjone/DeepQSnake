import torch

class SnakeBrain(torch.nn.Module):
    def __init__(self, output: int):
        super(SnakeBrain, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.conv2 = torch.nn.Conv2d(16, 32, 3)
        self.conv3 = torch.nn.Conv2d(32, 16, 3)
        self.fc1 = torch.nn.Linear(256, output)

        self.relu = torch.nn.ReLU()
        
        
    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        return x