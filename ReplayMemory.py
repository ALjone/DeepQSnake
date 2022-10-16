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
            self.apple_memory.append(Transition(*args[:-1]))
        else:
            self.other_memory.append(Transition(*args[:-1]))

    def sample(self, batch_size) -> List[Transition]:
        #NOTE: Lots of hard coded values to beware of
        samples = []
        apple_memory_sample_size = batch_size//2
        samples += random.sample(self.apple_memory, apple_memory_sample_size)
        samples += random.sample(self.other_memory, batch_size-apple_memory_sample_size)
        random.shuffle(samples)
        return samples



    def __len__(self):
        return min(len(self.apple_memory)*2, len(self.other_memory)*2)