import torch
from torch import nn


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
    def __init__(self, input_size: int, output: int, noise: float, frame_stack:int, device):
        super(SnakeBrain, self).__init__()

        #Part 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3*frame_stack, 32, 3, padding = (3, 3)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(64, 64, 2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
        )

        conv_output = 2304#((input_size-3)**2)*32*3

        self.linear = nn.Sequential(
            NoiseLayer(conv_output, 1024, noise, device),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.1),
            NoiseLayer(1024, 512, noise, device),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            NoiseLayer(512, 256, noise, device),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1)
        )

        #Value of state
        self.V = nn.Linear(256, 1)

        #Impact of action
        self.A = nn.Linear(256, output)

    def forward(self, x: torch.Tensor, return_separate = False):
        if len(x.shape) == 3:
            x = x.unsqueeze(0)
        x = x.float()
        x = torch.flatten(self.conv1(x), 1)
        x = self.linear(x)
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