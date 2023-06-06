import re
from symbol import term
import torch
from torch import nn
import random

from Transition import Transition

class Policy:
    def __init__(self,epsilon,epsilon_decay,learning_rate):
        super().__init__()
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
        print(f"Using {self.device} device")

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.flatten = nn.Flatten()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.model_stack = nn.Sequential(
            nn.Linear(8,128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.model_stack(x)
    
    def train(self, train_data : Transition):

        

        # forward pass
        prediction = self.model_stack(state).gather(1, action)

        # backward pass
        with torch.no_grad():
            target = reward + terminal * self.model_stack(next_state).max(1)[0].view(-1, 1)
        loss = self.loss_fn(prediction, target)

        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


    def save(self, path):
        torch.save(self.model_stack.state_dict(), path)

    def load(self, path):
        self.model_stack.load_state_dict(torch.load(path))

    def select_action(self, state):
        # epilson greedy policy
        if torch.rand(1) < self.epsilon:
            return torch.tensor([[random.randrange(4)]], device=self.device, dtype=torch.long)
        else:
            with torch.no_grad():
                return self.model_stack(state).max(1)[1].view(1, 1) 

    def decay(self):
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay


