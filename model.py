import torch

class SnakeBrain(torch.nn.Module):
    def __init__(self, output: int, extra: int = 1):
        super(SnakeBrain, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, 3)
        self.conv2 = torch.nn.Conv2d(16, 16, 3)
        self.conv3 = torch.nn.Conv2d(16, 8, 3)
        self.fc1 = torch.nn.Linear(128+extra, 64)
        self.fc2 = torch.nn.Linear(64, 32)
        self.fc3 = torch.nn.Linear(32, output)
        self.relu = torch.nn.ReLU()
        
        
    def forward(self, x: torch.Tensor, extra: float):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        if type(extra) == float:
            feature = torch.zeros(1).unsqueeze(0)
            feature[0] = extra
        else: feature = extra.unsqueeze(1)
        x = torch.cat((x, feature), dim=1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x  

    def load_model(self, state_dict):
        with torch.no_grad():
            self.conv1.weight.copy_(state_dict['conv1.weight'])
            self.conv1.bias.copy_(state_dict['conv1.bias'])
            self.conv2.weight.copy_(state_dict['conv2.weight'])
            self.conv2.bias.copy_(state_dict['conv2.bias'])
            self.conv3.weight.copy_(state_dict['conv3.weight'])
            self.conv3.bias.copy_(state_dict['conv3.bias'])