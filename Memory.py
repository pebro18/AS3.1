from Transition import Transition
from collections import deque
import random

class Memory:
    def __init__(self) -> None:
        self.memory = deque(maxlen=32000)
    
    def store(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size : int = 32):
        return random.sample(self.memory, batch_size)
