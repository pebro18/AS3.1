import random
import torch
from torch import nn

class Policy:
    def __init__(self,epsilon = 1,epsilon_decay = 0.99,learning_rate = 0.001):
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

        self.model_stack = nn.Sequential(
            nn.Linear(8,128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4)
        ).to(self.device)

        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model_stack.parameters(), lr=self.learning_rate)

    def forward(self, x):
        tensor_x = torch.Tensor(x).to(self.device)
        
        self.model_stack.eval()
        output = self.model_stack(tensor_x).to(self.device)
        self.model_stack.train()
        return output
    
    def init_model_with_random_weights(self):
        for layer in self.model_stack:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()


    def save(self, path):
        torch.save(self.model_stack.state_dict(), path)

    def load(self, path):
        self.model_stack.load_state_dict(torch.load(path))

    def model_train(self, train_loader):

        for batch, (X, y) in enumerate(train_loader):
            self.optimizer.zero_grad()
            loss = self.loss_fn(X, y)
            loss.backward()
            self.optimizer.step()

        return loss.item()

    def select_action(self, state):
        # epilson greedy policy
        if random.random() < self.epsilon:
            return torch.rand(4, device=self.device).max(0)[1].view(1, 1).item()
        else:
            with torch.no_grad():
                return self.forward(state).max(0)[1].view(1, 1).item()

    def decay(self):
        if self.epsilon > 0.01:
            self.epsilon *= self.epsilon_decay



