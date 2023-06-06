
import torch 

from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose

print(torch.cuda.is_available())
print(torch.cuda.device_count())

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class gymnasium():
    import gymnasium as gym
    
    env = gym.make("LunarLander-v2", render_mode="human")
    observation, info = env.reset(seed=42)
    for _ in range(1000):
        action = env.action_space.sample()  # this is where you would insert your policy
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
        env.close()


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.learning_rate = 0.001
        self.flatten = nn.Flatten()
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        self.model_stack = nn.Sequential(
            nn.Linear(8,128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )

    def forward(self, x):
        x = self.flatten(x)
        return self.model_stack(x)
    
    def train(self, train_loader):
        for batch, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            self.optimizer.zero_grad()
            output = self.model_stack(data)
            loss = self.loss_fn(output, target)
            loss.backward()
            self.optimizer.step()
            if batch % 100 == 0:
                print(f"loss: {loss.item()} batch: {batch}")

    def test(self, test_loader):

        size = len(test_loader.dataset)
        test_loss, correct = 0, 0

        with torch.no_grad():
            for batch, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                pred = self.model_stack(data)
                test_loss += self.loss_fn(pred, target).item()
                correct += (pred.argmax(1) == target).type(torch.float).sum().item()
        
        test_loss /= size
        correct /= size
        print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def save(self, path):
        torch.save(self.model_stack.state_dict(), path)

