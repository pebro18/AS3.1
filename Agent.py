import gymnasium as gym
from Policy import Policy
from Memory import Memory
from Transition import Transition

class Agent():
    def init(self,policy: Policy, memory: Memory, discount_factor: float = 1, amount_steps: int = 2000, memory_batch_size: int = 32):
        self.env = gym.make("LunarLander-v2", render_mode="human")	
        self.policy = policy
        self.memory = memory

        self.discount_factor = discount_factor
        self.amount_steps = amount_steps
        self.memory_batch_size = memory_batch_size


    def run(self):
        observation, info = self.env.reset()
        for _ in range(2000):
            action = self.policy.select_action(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)
            if self.terminated or self.truncated:
                observation, info = self.env.reset()
        self.env.close()
        
    def train(self,epochs: int):
        
        self.memory.init_memory()
        self.policy.init_model_with_random_weights()

        for epoch in range(epochs):
            observation, info = self.env.reset()
            for _ in range(self.amount_steps):
                action = self.policy.select_action(observation)
                observation_prime, reward, terminated, truncated, info = self.env.step(action)
                if self.terminated or self.truncated:
                    observation, info = self.env.reset()
                    continue

                self.memory.store(Transition(observation, action, reward, observation_prime, terminated))
                memory_batch = self.memory.sample()
                target_batch = []
                for transition in memory_batch:
                    target = None
                    
                    action_prime = self.policy.forward(transition.next_state)
                    if transition.terminal:
                        target = transition.reward
                    else:
                        target = transition.reward + self.discount_factor * action_prime
                    target_batch.append(target)
                self.policy.train(memory_batch,target_batch)
                observation = observation_prime
            self.policy.decay()

        pass

