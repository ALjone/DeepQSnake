import torch
from torch import nn
import numpy as np


class NoiseLayer(nn.Module):
    def __init__(self, input: int, output: int, noise: float, device) -> None:
        super(NoiseLayer, self).__init__()
        self.device = device
        #https://arxiv.org/pdf/1706.10295.pdf
        self.noisy_part = nn.Linear(input, output) 
        self.noisy_part.weight = nn.Parameter(torch.ones(self.noisy_part.weight.shape)*0.017)
        self.noisy_part.bias = nn.Parameter(torch.ones(self.noisy_part.bias.shape)*0.017)
        self.noise_std = noise


        self.linear = nn.Linear(input, output)

    def forward(self, x):
        noise = self.get_noise(x)
        return self.linear(x)+noise

    def get_noise(self, x):
        noise_w = torch.normal(0, self.noise_std, size = self.noisy_part.weight.shape).to(self.device)
        noise_b = torch.normal(0, self.noise_std, size = self.noisy_part.bias.shape).to(self.device)

        W = (self.noisy_part.weight * noise_w)
        B = (self.noisy_part.bias*noise_b).unsqueeze(1)

        return (B + (W@x.T)).T

    def get_noise_level(self):
        return torch.sum(torch.abs(self.noisy_part.weight))/torch.sum(self.noisy_part.weight.shape)


class SnakeBrain(torch.nn.Module):
    def __init__(self, input_size: int, output: int, noise: float, frame_stack: int, device):
        super(SnakeBrain, self).__init__()

        #Part 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3*frame_stack, 16, 2, padding = (1, 1)),
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
        self.conv2 = nn.Sequential(
        nn.Conv2d(3*frame_stack, 16, 3, padding = (2, 2)),
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

        #Part 3
        self.conv3 = nn.Sequential(
        nn.Conv2d(3*frame_stack, 16, 4, padding = (3, 3)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 16, 2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )

        conv_output = ((input_size-3)**2)*16*3

        #Value of state
        self.V = nn.Sequential(
            NoiseLayer(conv_output, 256, noise, device),
            nn.BatchNorm1d(256),
            nn.ReLU(),   
            NoiseLayer(256, 128, noise, device),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            NoiseLayer(128, 64, noise, device),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            NoiseLayer(64, 1, noise, device)
        )   
        #Impact of action
        self.A = nn.Sequential(
            NoiseLayer(conv_output, 256, noise, device),
            nn.BatchNorm1d(256),
            nn.ReLU(),   
            NoiseLayer(256, 128, noise, device),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            NoiseLayer(128, 64, noise, device),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            NoiseLayer(64, output, noise, device)
        )   
        
        
    def forward(self, x: torch.Tensor, return_separate = False):
        if type(x) == np.ndarray:
            x = torch.tensor(x)
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()
        x = torch.concat((torch.flatten(self.conv1(x), 1), torch.flatten(self.conv2(x), 1), torch.flatten(self.conv3(x), 1)), axis = 1)
        A = self.A(x)
        V = self.V(x)
        if return_separate:
            return V + (A-A.mean(dim = 1, keepdim = True)), V, A
        return V + (A-A.mean(dim = 1, keepdim = True))



    def get_noise_level(self):
        noise = 0
        i = 0
        for n, p in self.named_parameters():
            if "noisy" in n and "weight" in n:
                noise += torch.sum(torch.abs(p))/p.nelement()
                i += 1
        return (noise/i).item()