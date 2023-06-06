import gymnasium as gym

from Policy import Policy
from Memory import Memory

class Agent():
    def init(self,policy: Policy, memory: Memory):
        self.env = gym.make("LunarLander-v2", render_mode="human")	
        self.observation, self.info = self.env.reset(seed=42)
        self.policy = policy
        self.memory = memory

    def run(self):
        for _ in range(2000):
            action = self.policy.select_action(self.observation)
            self.observation, reward, terminated, truncated, self.info = self.env.step(action)
            if self.terminated or self.truncated:
                observation, info = self.env.reset()
        self.env.close()
        
    def train(self,):
        memory_batch = self.memory.sample()
        for transition in memory_batch:
            self.policy.train(transition)

        pass

