import numpy as np
import torch
from typing import List

class Memory:
    def __init__(self) -> None:
        self.state = None
        self.next_state = None
        self.reward = None
        self.action = None
        self.done = False


class MemoryBank:
    def __init__(self, capacity: int = 500) -> None:
        self.filled: bool = False
        self.capacity: int = capacity
        self.memories: list(Memory) = [0] * self.capacity
        self.index: int = 0

    def addMemory(self, memory: Memory) -> None:
        self.filled = True if self.index == self.capacity-1 else self.filled
        self.memories[self.index] = memory
        self.index = (self.index + 1) % self.capacity


    def getSamples(self, batch_size: int) -> List[Memory]:
        return [
            self.memories[i] for i in np.random.choice(self.capacity-1 if self.filled else self.index-1, batch_size, )
        ]
