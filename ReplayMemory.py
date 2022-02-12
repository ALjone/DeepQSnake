#Borrowed from https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import random
from collections import namedtuple
from collections import deque
from typing import List

class Memory:
    def __init__(self) -> None:
        self.state = None
        self.next_state = None
        self.reward = None
        self.action = None
        self.done = False

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """The base for this class is copied from """
    def __init__(self, capacity, apple_reward, death_reward):
        self.apple_memory = deque([],maxlen=capacity)
        self.death_memory = deque([],maxlen=capacity)
        self.other_memory = deque([], maxlen=capacity*2)
        self.apple_reward = apple_reward
        self.death_reward = death_reward

    def push(self, *args):
        if args[3] == self.apple_reward:
            self.apple_memory.append(Transition(*args))
        elif args[3] == self.death_reward:
            self.death_memory.append(Transition(*args))
        else:
            self.other_memory.append(Transition(*args))

    def sample(self, batch_size) -> List[Transition]:
        samples = []
        sample_size = batch_size//4
        samples += random.sample(self.apple_memory, sample_size)
        samples += random.sample(self.death_memory, sample_size)
        samples += random.sample(self.other_memory, batch_size-(sample_size*2))
        random.shuffle(samples)
        return samples

    def __len__(self):
        return min(len(self.apple_memory), len(self.death_memory), len(self.other_memory))