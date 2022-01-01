import torch

class SnakeBrain(torch.nn.Module):
    def __init__(self, output: int, extra: int = 1):
        super(SnakeBrain, self).__init__()
        self.extra = extra
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.conv2 = torch.nn.Conv2d(16, 16, 3)
        self.conv3 = torch.nn.Conv2d(16, 8, 3)
        self.fc1 = torch.nn.Linear(128+extra, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, output)
        self.relu = torch.nn.ReLU()
        
        
    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        feature = torch.ones(x.shape[0], self.extra)
        x = torch.cat((x, feature), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x  