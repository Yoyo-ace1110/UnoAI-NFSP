import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)    # 全連接層  輸入=>256
        self.fc2 = nn.Linear(256, 256)          # 隱藏層    256=>256
        self.out = nn.Linear(256, action_dim)   # 輸出層

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity: int = 10000) -> None:
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float, next_state, done: bool) -> None:
        """ 將一步遊戲經驗存入記憶庫 """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """ 隨機抽樣一個批次進行訓練 """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32)
        )

    def __len__(self) -> int:
        return len(self.buffer)
