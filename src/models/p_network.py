import os
import random
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from core import Card

class PNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, file_path: str = "p_network.bin") -> None:
        """ Policy Network """
        super().__init__()
        self.file_path: str = file_path
        self.fc1 = nn.Linear(state_dim, 256)    # 全連接層  輸入=>256
        self.fc2 = nn.Linear(256, 256)          # 隱藏層    256=>256
        self.out = nn.Linear(256, action_dim)   # 輸出層    256=>輸出

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Forward Pass """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

    def save(self) -> None:
        """ 儲存的權重檔 """
        torch.save(self.state_dict(), self.file_path)
        print(f"Policy-Network 權重已成功儲存至: {self.file_path}")
    
    def load(self) -> None:
        """ 讀取權重檔 """
        if os.path.exists(self.file_path):
            self.load_state_dict(torch.load(self.file_path))
            print(f"已成功載入 Policy-Network 權重: {self.file_path}")
        else: print(f"找不到權重檔 {self.file_path}。")

class ReservoirMemory:
    def __init__(self, file_path: str = "ReservoirMemory.bin", capacity: int = 10000) -> None:
        """ 蓄水池記憶庫 """
        self.file_path: str = file_path
        self.buffer: list[tuple[torch.Tensor, int]] = []
        self.seen_count: int = 0        # 包含被淘汰的舊資料
        self.capacity: int = capacity   # 單純目前記憶庫容量

    def load(self) -> None:
        """ 讀取二進位訓練資料 """
        if os.path.exists(self.file_path):
            with open(self.file_path, mode="rb") as file:
                self.buffer = pickle.load(file=file)
            print(f"ReservoirMemory 已成功從 {self.file_path} 讀取")
        else: print(f"無法從 {self.file_path} 讀取 ReservoirMemory ")

    def save(self) -> None:
        """ 儲存二進位訓練資料 """ 
        with open(self.file_path, mode="wb") as file:
            pickle.dump(self.buffer, file=file)
        print(f"ReplayBuffer 已成功儲存至: {self.file_path}")

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
        return (torch.stack(states), torch.tensor(actions, dtype=torch.int64))

class PolicyTrainer:
    def __init__(self, state_dim: int, action_dim: int = 61, batch_size: int = 64) -> None:
        self.batch_size = batch_size
        self.memory = ReservoirMemory()
        self.p_net = PNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.p_net.parameters(), lr=0.001)
        # 多分類問題使用 CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss()
        
        self.steps_since_update = 0

    def push(self, state: torch.Tensor, card: Card|None) -> float | None:
        """ 每次產生行為經驗時呼叫：存入蓄水池，並檢查是否達標觸發訓練 """
        self.memory.push(state, card)
        self.steps_since_update += 1
        # 達標且蓄水池資料足夠時觸發
        if self.steps_since_update >= self.batch_size and self.memory.size() >= self.batch_size:
            self.steps_since_update = 0
            return self.update()
        return None

    def update(self) -> float:
        """ 執行一次 Policy Network 的監督學習更新 """
        states, actions = self.memory.sample(self.batch_size)
        # 透過 Policy Network 預測動作的分數 (Logits)
        logits = self.p_net(states)  # shape: (batch_size, action_dim)
        # 計算交叉熵損失 (比較預測的動作分佈與實際執行的 action)
        loss = self.criterion(logits, actions)
        # 反向傳播與優化更新
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step() # pyright: ignore[reportUnknownMemberType]
        return loss.item()

    def save(self) -> None:
        """ 儲存的訓練資料 """
        self.memory.save()
        self.p_net.save()

    def load(self) -> None:
        """ 讀取權重檔 """
        self.memory.load()
        self.p_net.load()
