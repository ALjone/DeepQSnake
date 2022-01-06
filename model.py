import torch

class SnakeBrain(torch.nn.Module):
    def __init__(self, input_size: int, output: int):
        super(SnakeBrain, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 8, 2)
        self.conv2 = torch.nn.Conv2d(8, 16, 2)
        self.conv3 = torch.nn.Conv2d(16, 32, 2)
        self.conv4 = torch.nn.Conv2d(32, 16, 3)
        self.conv5 = torch.nn.Conv2d(16, 8, 3)
        conv_output = ((input_size-7)**2)*8
        self.fc1 = torch.nn.Linear(conv_output, 128)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, output)

        self.relu = torch.nn.ReLU()        
        
    def forward(self, x: torch.Tensor):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = torch.flatten(x, 1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x  
