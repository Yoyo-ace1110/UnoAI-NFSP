import os
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
from torch import Tensor
from core import Card

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, file_path: str = "q_network_v1.bin") -> None:
        """ Quality Network """
        super().__init__()
        self.file_path: str = file_path
        self.fc1: nn.Linear = nn.Linear(state_dim, 256)    # 全連接層  輸入=>256
        self.fc2: nn.Linear = nn.Linear(256, 256)          # 隱藏層    256=>256
        self.out: nn.Linear = nn.Linear(256, action_dim)   # 輸出層    256=>輸出

    def forward(self, x: Tensor) -> Tensor:
        """ Forward Pass """
        x_: Tensor = x
        x_: Tensor = F.relu(self.fc1(x_))
        x_: Tensor = F.relu(self.fc2(x_))
        return self.out(x_)

    def save(self) -> None:
        """ 儲存的權重檔 """
        torch.save(self.state_dict(), self.file_path)
        print(f"Q-Network 權重已成功儲存至: {self.file_path}")
    
    def load(self) -> None:
        """ 讀取權重檔 """
        if os.path.exists(self.file_path):
            self.load_state_dict(torch.load(self.file_path))
            print(f"已成功載入 Q-Network 權重: {self.file_path}")
        else: print(f"找不到權重檔 {self.file_path}。")

class ReplayBuffer:
    def __init__(self, file_path: str = "ReplayBuffer_v1.bin", capacity: int = 10000) -> None:
        """ Q-Network 記憶庫 """
        self.buffer: deque[tuple[Tensor, int, float, Tensor, bool]]
        self.buffer = deque(maxlen=capacity)
        self.file_path: str = file_path

    def load(self) -> None:
        """ 讀取二進位訓練資料 """
        if os.path.exists(self.file_path):
            with open(self.file_path, mode="rb") as file:
                self.buffer = pickle.load(file=file)
            print(f"ReplayBuffer 已成功從 {self.file_path} 讀取")
        else: print(f"無法從 {self.file_path} 讀取 ReplayBuffer ")

    def save(self) -> None:
        """ 儲存二進位訓練資料 """ 
        with open(self.file_path, mode="wb") as file:
            pickle.dump(self.buffer, file=file)
        print(f"ReplayBuffer 已成功儲存至: {self.file_path}")

    def size(self) -> int: return len(self.buffer)

    def push(self, state: Tensor, card: Card|None, reward: float, next_state: Tensor, done: bool) -> None:
        """ 將一步遊戲經驗存入記憶庫 (FIFO) """
        action: int = (60 if card is None else card.id)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> tuple[Tensor, ...]:
        """ 隨機抽樣一個批次進行訓練 """
        batch: list[tuple[Tensor, int, float, Tensor, bool]]
        batch = random.sample(self.buffer, batch_size)
        states:         tuple[Tensor, ...]
        actions:        tuple[int, ...]
        rewards:        tuple[float, ...]
        next_states:    tuple[Tensor, ...]
        dones:          tuple[bool, ...]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.stack (states),
            torch.tensor(actions, dtype=torch.int64),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack (next_states),
            torch.tensor(dones, dtype=torch.float32)
        )

class DQNTrainer:
    def __init__(self, state_dim: int, action_dim: int = 61, batch_size: int = 64, gamma: float = 0.99) -> None:
        """ Deep Q-Network Trainer """
        self.gamma: float = gamma
        self.batch_size: int = batch_size
        self.memory: ReplayBuffer = ReplayBuffer()
        self.q_net: QNetwork = QNetwork(state_dim, action_dim)
        # 使用 Adam 優化器
        self.optimizer: optim.Adam = optim.Adam(self.q_net.parameters(), lr=0.001)
        self.criterion: nn.MSELoss = nn.MSELoss()  # 均方誤差損失
        # 用來計算增加多少筆資料的計數器
        self.steps_since_update: int = 0

    def push(self, state: Tensor, card: Card|None, reward: float, next_state: Tensor, done: bool) -> float|None:
        """ 存入記憶庫，並檢查是否達到 batch_size 觸發 update """
        self.steps_since_update += 1
        self.memory.push(state, card, reward, next_state, done)
        # 當累積新增的資料量達到 batch_size 且記憶庫有足夠資料時觸發訓練
        if (self.steps_since_update >= self.batch_size and self.memory.size() >= self.batch_size):
            self.steps_since_update = 0  # 重置計數器
            return self.update()
        return None

    def update(self) -> float:
        """ 執行一次 DQN 梯度下降更新 """
        states:         Tensor
        actions:        Tensor
        rewards:        Tensor
        next_states:    Tensor
        dones:          Tensor
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        # 計算當前狀態的 Q 值
        q_values: Tensor = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # 計算目標 Q 值 (Bellman Equation)
        with torch.no_grad():
            # max(1)[0] 取得下一個狀態中所有動作的最大 Q 值
            next_q_values: Tensor = self.q_net(next_states).max(1)[0]
            # 若 done 為 True，則沒有未來回報 (1 - dones)
            target_q_values: Tensor = rewards + self.gamma*next_q_values*(1.0-dones)
        # 計算 Loss
        loss: Tensor = self.criterion(q_values, target_q_values)
        # 反向傳播與優化更新
        self.optimizer.zero_grad()
        loss.backward()       # pyright: ignore[reportUnknownMemberType]
        self.optimizer.step() # pyright: ignore[reportUnknownMemberType]
        return float(loss.item())

    def save(self) -> None:
        """ 儲存的訓練資料 """
        self.memory.save()
        self.q_net.save()

    def load(self) -> None:
        """ 讀取權重檔 """
        self.memory.load()
        self.q_net.load()
