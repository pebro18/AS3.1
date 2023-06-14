from datetime import datetime
import numpy as np
import torch
from tqdm import tqdm
import gymnasium as gym
from collections import deque
import matplotlib.pyplot as plt
from copy import deepcopy

from Policy import Policy
from Memory import Memory
from Transition import Transition


class Agent:
    def init(
        self,
        policy: Policy,
        memory: Memory,
        discount_factor: float = 0.99,
        amount_steps: int = 2000,
        memory_batch_size: int = 32,
        tau: float = 0.08,
    ):
        self.env = gym.make("LunarLander-v2", render_mode="rgb_array")
        self.policy_control = deepcopy(policy)
        self.policy_target = deepcopy(policy)
        self.memory = memory

        self.discount = discount_factor
        self.amount_steps = amount_steps
        self.memory_batch_size = memory_batch_size
        self.tau = tau

    def run(self, model_path: str):
        self.policy_control.load(model_path)
        self.policy_control.model_stack.eval()
        self.policy_control.epsilon = 0.0

        self.env = gym.make("LunarLander-v2", render_mode="human")
        observation, info = self.env.reset()
        for _ in range(20000):
            action = self.policy_control.select_action(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)
            if terminated or truncated:
                observation, info = self.env.reset()
        self.env.close()

    def train(self, epochs: int):
        scores = []
        scores_window = deque(maxlen=100)


        for epoch in tqdm(range(epochs), desc=f"Epochs: {epochs}", unit="epoch"):
            observation, info = self.env.reset()
            score = 0
            for _ in tqdm(
                range(self.amount_steps), desc=f"Epoch: {epoch}", unit="step"
            ):
                # act
                action = self.policy_control.select_action(observation)
                # observe
                observation_prime, reward, terminated, truncated, info = self.env.step(
                    action
                )
                # store
                self.memory.store(
                    Transition(
                        observation, action, reward, observation_prime, terminated
                    )
                )

                observation = observation_prime
                score += reward

                # learn
                self.learn()
                # align
                self.align_target_model()

                if terminated or truncated:
                    break
            scores_window.append(score)
            scores.append(score)
            print(
                f"Epoch: {epoch} Score: {score} Average Score: {np.average(scores_window)}"
            )

            if epoch % 50 == 0:
                self.policy_control.save(
                    f"./Model_outputs/checkpoints/epochs_{epoch}_checkpoint.pt"
                )

            if np.average(scores_window) >= 200:
                print(
                    f"Environment solved in {epoch} episodes! Average Score: {np.average(scores_window)}"
                )
                break
            self.policy_control.decay()
        self.env.close()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.policy_control.save(f"./Model_outputs/epochs_{epochs}_{timestamp}.pt")
        self.plot_scores(scores)

    def learn(self):
        if len(self.memory.memory) < self.memory_batch_size:
            return

        memory_batch = self.memory.sample()

        expected_q_batch = []
        target_q_batch = []

        for transition in memory_batch:      
            pred_q_values = self.policy_control.forward(transition.state)
            pred_action = pred_q_values[transition.action]

            target_q_values = self.policy_target.forward(transition.next_state)
            target_q_values = torch.max(target_q_values) * (1 - transition.terminal)

            y_j = transition.reward + self.discount * target_q_values

            expected_q_batch.append(pred_action)
            target_q_batch.append(y_j)

        expected_q_batch = torch.stack(expected_q_batch)
        target_q_batch = torch.stack(target_q_batch)
        self.policy_control.model_train(expected_q_batch, target_q_batch)


    # Way faster method of learning the model
    # this is meant to be a test/optimization method i found on github
    def batch_learn(self):

        if len(self.memory.memory) < self.memory_batch_size:
            return

        memory_batch = self.memory.sample()
        states = [transition.state for transition in memory_batch]
        actions = [transition.action for transition in memory_batch]
        rewards = [transition.reward for transition in memory_batch]
        next_states = [transition.next_state for transition in memory_batch]
        terminals = [transition.terminal for transition in memory_batch]


        pred_q_values = self.policy_control.forward(states)
        pred_q_values = pred_q_values.gather(1, torch.tensor(actions,device=self.policy_control.device).unsqueeze(-1)).squeeze(-1)

        target_q_values = self.policy_target.forward(next_states)
        target_q_values = target_q_values.max(1)[0]
        target_q_values[terminals] = 0.0

        y_js = torch.tensor(rewards,device=self.policy_control.device).to(torch.float32) + self.discount * target_q_values
        self.policy_control.model_train(pred_q_values, target_q_values)


    def plot_scores(self, scores):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel("Score")
        plt.xlabel("Epochs #")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"./Model_outputs/plot_{len(scores)}_{timestamp}.png")
        plt.show()

    def align_target_model(self):
        for target_param, param in zip(
            self.policy_target.model_stack.parameters(),
            self.policy_control.model_stack.parameters(),
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )
