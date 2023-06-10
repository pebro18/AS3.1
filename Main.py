import gymnasium as gym

from Policy import Policy
from Memory import Memory
from Agent import Agent


# env = gym.make("LunarLander-v2", render_mode="human")
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#    action = env.action_space.sample()  # this is where you would insert your policy
#    observation, reward, terminated, truncated, info = env.step(action)

#    if terminated or truncated:
#       observation, info = env.reset()
# env.close()


Agent = Agent()
Agent.init(Policy(),Memory())
Agent.train(200)
# Agent.run()