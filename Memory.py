from Transition import Transition
from collections import deque
import random

class Memory:
    def __init__(self,capacity = 32000) -> None:
        self.memory = deque(maxlen=capacity)
    
    def init_memory(self):

        
        pass

    def store(self, transition: Transition):
        self.memory.append(transition)

    def sample(self, batch_size : int = 32):
        return random.sample(self.memory, batch_size)
