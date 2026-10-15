import random
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from core import Card

class PNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        """ Policy Network """
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 256)    # 全連接層  輸入=>256
        self.fc2 = nn.Linear(256, 256)          # 隱藏層    256=>256
        self.out = nn.Linear(256, action_dim)   # 輸出層    256=>輸出

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward Pass """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class ReservoirMemory:
    def __init__(self, file_path: str = "p_network.bin", capacity: int = 10000) -> None:
        """ 蓄水池記憶庫 """
        self.file_path: str = file_path
        self.buffer: list[tuple[torch.Tensor, int]] = []
        self.seen_count: int = 0        # 包含被淘汰的舊資料
        self.capacity: int = capacity   # 單純目前記憶庫容量

    def load(self) -> None:
        """ 讀取二進位訓練資料 """
        with open(self.file_path, mode="rb") as file:
            self.buffer = pickle.load(file=file)

    def save(self) -> None:
        """ 儲存二進位訓練資料 """ 
        with open(self.file_path, mode="wb") as file:
            pickle.dump(self.buffer, file=file)

    def size(self) -> int: return len(self.buffer)

    def push(self, state: torch.Tensor, card: Card|None) -> None:
        """ Reservoir Sampling (蓄水池抽樣) """
        self.seen_count += 1
        action: int = (60 if card is None else card.id)
        data: tuple[torch.Tensor, int] = (state, action)
        # 記憶庫未滿則直接插入新資料
        if (len(self.buffer) < self.capacity): return self.buffer.append(data)
        # 新資料以遞減機率隨機替換掉舊資料
        index = random.randint(0, self.seen_count-1)
        if (index < self.capacity): self.buffer[index] = data

    def sample(self, batch_size: int):
        """ 隨機抽樣一個批次進行監督學習訓練 """
        batch = random.sample(self.buffer, batch_size)
        states, actions = zip(*batch)
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64)
        )
