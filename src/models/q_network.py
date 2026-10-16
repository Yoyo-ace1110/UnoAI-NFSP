import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
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

class DQNTrainer:
    def __init__(self, state_dim: int, action_dim: int = 61, batch_size: int = 64, gamma: float = 0.99) -> None:
        """ Deep Q-Network Trainer """
        self.gamma = gamma
        self.batch_size = batch_size
        self.q_net = QNetwork(state_dim, action_dim)
        self.memory = ReplayBuffer()
        # 使用 Adam 優化器
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()  # 均方誤差損失
        # 用來計算增加多少筆資料的計數器
        self.steps_since_update = 0

    def push(self, state: torch.Tensor, card: Card|None, reward: float, next_state: torch.Tensor, done: bool) -> float|None:
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
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        # 計算當前狀態的 Q 值
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        # 計算目標 Q 值 (Bellman Equation)
        with torch.no_grad():
            # max(1)[0] 取得下一個狀態中所有動作的最大 Q 值
            next_q_values = self.q_net(next_states).max(1)[0]
            # 若 done 為 True，則沒有未來回報 (1 - dones)
            target_q_values = rewards + self.gamma*next_q_values*(1.0-dones)
        # 計算 Loss (預測值與目標值的差距)
        loss = self.criterion(q_values, target_q_values)
        # 反向傳播與優化更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
