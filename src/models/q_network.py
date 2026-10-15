import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from core import Card

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        """ Quality Network """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)    # 全連接層  輸入=>256
        self.fc2 = nn.Linear(256, 256)          # 隱藏層    256=>256
        self.out = nn.Linear(256, action_dim)   # 輸出層    256=>輸出

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward Pass """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, file_path: str = "q_network.bin", capacity: int = 10000) -> None:
        """ Q-Network 記憶庫 """
        self.buffer: deque[tuple[torch.Tensor, int, float, torch.Tensor, bool]]
        self.buffer = deque(maxlen=capacity)
        self.file_path: str = file_path

    def load(self) -> None:
        """ 讀取二進位訓練資料 """
        with open(self.file_path, mode="rb") as file:
            self.buffer = pickle.load(file=file)

    def save(self) -> None:
        """ 儲存二進位訓練資料 """ 
        with open(self.file_path, mode="wb") as file:
            pickle.dump(self.buffer, file=file)

    def size(self) -> int: return len(self.buffer)

    def push(self, state: torch.Tensor, card: Card|None, reward: float, next_state: torch.Tensor, done: bool) -> None:
        """ 將一步遊戲經驗存入記憶庫 (FIFO) """
        action: int = (60 if card is None else card.id)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
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
