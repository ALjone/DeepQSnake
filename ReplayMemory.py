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
    def __init__(self, capacity):
        self.apple_memory = deque([],maxlen=capacity)
        self.death_memory = deque([],maxlen=capacity)
        self.towards_memory = deque([],maxlen=capacity)
        self.away_memory = deque([],maxlen=capacity)

    def push(self, *args):
        if args[3] == 1.0:
            self.apple_memory.append(Transition(*args))
        elif args[3] == -1.0:
            self.death_memory.append(Transition(*args))
        elif args[3] == 0.2:
            self.towards_memory.append(Transition(*args))
        elif args[3] == -0.2:
            self.away_memory.append(Transition(*args))
        else:
            print("Something went wrong, got a reward of", args[3])

    def sample(self, batch_size) -> List[Transition]:
        samples = []
        sample_size = batch_size//4
        samples += random.sample(self.apple_memory, sample_size)
        samples += random.sample(self.death_memory, sample_size)
        samples += random.sample(self.towards_memory, sample_size)
        samples += random.sample(self.away_memory, batch_size-(sample_size*3))
        random.shuffle(samples)
        return samples

    def __len__(self):
        return min(len(self.apple_memory), len(self.death_memory), len(self.towards_memory), len(self.away_memory))*4