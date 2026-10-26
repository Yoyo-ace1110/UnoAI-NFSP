import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .device import DEVICE, to_device
from torch import Tensor
from core import Card

# 60 種牌 + 抽牌
NUM_ACTIONS: int = 61

class PNetwork(nn.Module):
    def __init__(self, state_dim: int, file_path: str = "p_network_v1.bin") -> None:
        """ Policy Network """
        super().__init__()
        self.file_path: str = file_path
        self.fc1: nn.Linear = nn.Linear(state_dim, 256)   # 全連接層  輸入=>256
        self.fc2: nn.Linear = nn.Linear(256, 256)         # 隱藏層    256=>256
        self.fc3: nn.Linear = nn.Linear(256, 256)         # 隱藏層    256=>256
        self.out: nn.Linear = nn.Linear(256, NUM_ACTIONS) # 輸出層    256=>輸出
        self.to(DEVICE)

    def forward(self, x: Tensor) -> Tensor:
        """ Forward Pass """
        x_: Tensor = x
        x_ = F.relu(self.fc1(x_))
        x_ = F.relu(self.fc2(x_))
        x_ = F.relu(self.fc3(x_))
        return self.out(x_)

    def save(self) -> None:
        """ 儲存的權重檔 """
        torch.save(self.state_dict(), self.file_path)
        print(f"Policy-Network 權重已成功儲存至: {self.file_path}")
    
    def load(self) -> bool:
        """ 讀取權重檔 回傳是否成功 """
        if (not os.path.exists(self.file_path)): 
            print(f"找不到權重檔 {self.file_path}")
            return False
        try:
            self.load_state_dict(torch.load(self.file_path, map_location=DEVICE, weights_only=True))
        except Exception as error:
            print(f"無法載入 {self.file_path} (可能是舊版的權重): {str(error).splitlines()[0]}")
            return False
        self.to(DEVICE)
        print(f"已成功載入 Policy-Network 權重: {self.file_path}")
        return True

class ReservoirMemory:
    def __init__(self, file_path: str = "ReservoirMemory_v1.bin", capacity: int = 64000) -> None:
        """ 蓄水池記憶庫 """
        self.file_path: str = file_path
        self.capacity: int = capacity   # 單純目前記憶庫容量
        self.seen_count: int = 0        # 包含被淘汰的舊資料
        self._size: int = 0
        self._masks:   Tensor|None = None
        self._states:  Tensor|None = None
        self._actions: Tensor|None = None

    def _alloc(self, state_dim: int) -> None:
        """ 預先配置張量的記憶體 """
        self._masks   = torch.ones ((self.capacity, NUM_ACTIONS), dtype=torch.bool)
        self._states  = torch.zeros((self.capacity, state_dim), dtype=torch.float32)
        self._actions = torch.zeros(self.capacity, dtype=torch.int64)

    def size(self) -> int: return self._size

    def push(self, state: Tensor, card: Card|None, mask: Tensor|None = None) -> None:
        """ Reservoir Sampling (蓄水池抽樣) """
        if (self._states is None): self._alloc(state.shape[0])
        assert (self._masks is not None)
        assert (self._states is not None)
        assert (self._actions is not None)
        self.seen_count += 1
        # 記憶庫未滿則直接插入新資料
        if (self._size < self.capacity): 
            index: int = self._size
            self._size += 1
        # 已滿則以遞減機率隨機替換掉舊資料
        else:
            index = random.randrange(self.seen_count)
            if (index >= self.capacity): return
        self._states[index]  = state
        self._actions[index] = (60 if (card is None) else card.id)
        self._masks[index]   = (True if (mask is None) else mask)

    def sample(self, batch_size: int) -> tuple[Tensor, Tensor, Tensor]:
        """ 隨機抽樣一個批次進行監督學習訓練: (states, actions, masks) """
        assert (self._masks is not None)
        assert (self._states is not None)
        assert (self._actions is not None)
        index: Tensor = torch.randint(self._size, (batch_size,))
        return (self._states[index], self._actions[index], self._masks[index])

    def save(self) -> None:
        """ 儲存二進位訓練資料 """ 
        if ((self._states is None) or (self._size == 0)):
            print(f"ReservoirMemory 是空的，略過 {self.file_path}")
            return
        assert (self._masks is not None)
        assert (self._actions is not None)
        n: int = self._size
        torch.save(
            {
                "masks":      self._masks[:n],
                "states":     self._states[:n], 
                "actions":    self._actions[:n], 
                "seen_count": self.seen_count,
            }, 
            self.file_path
        )
        print(f"ReservoirMemory 已成功儲存至: {self.file_path} ({n} 筆)")

    def load(self) -> bool:
        """ 讀取二進位訓練資料 """
        if (not os.path.exists(self.file_path)): 
            print(f"無法從 {self.file_path} 讀取 ReservoirMemory")
            return False
        try:
            data = torch.load(self.file_path, map_location="cpu", weights_only=True)  # type: ignore
            states: Tensor = data["states"]  # type: ignore
            n: int = min(int(states.shape[0]), self.capacity)
            self._alloc(int(states.shape[1]))
            assert (self._masks is not None)
            assert (self._states is not None)
            assert (self._actions is not None)
            self._states[:n]  = states[:n]
            self._actions[:n] = data["actions"][:n]  # type: ignore
            self._masks[:n]   = data["masks"][:n]    # type: ignore
            seen: int = int(data["seen_count"])      # type: ignore
        except Exception as error:
            print(f"無法讀取 {self.file_path} (舊版格式?)，略過: {str(error).splitlines()[0]}")
            self._states = None
            self.seen_count = 0
            self._size = 0
            return False
        self._size = n
        self.seen_count = max(seen, n)
        print(f"已成功從 {self.file_path} 讀取 ReservoirMemory ({n} 筆)")
        return True

class PolicyTrainer:
    def __init__(
        self, 
        state_dim: int, 
        lr: float = 1e-3, 
        batch_size: int = 64, 
        network_path: str = "p_network_v1.bin", 
        memory_path: str = "ReservoirMemory_v1.bin",
        update_every: int = 8,
        warmup: int = 1000,
    ) -> None:
        """ 這裡請補一段簡介 """
        self.batch_size: int = batch_size
        self.update_every: int = update_every
        self.warmup: int = max(warmup, batch_size)
        self.p_net: PNetwork = PNetwork(state_dim, network_path)
        self.memory: ReservoirMemory = ReservoirMemory(memory_path)
        self.optimizer: optim.Adam = optim.Adam(self.p_net.parameters(), lr)
        self.steps_since_update: int = 0

    def push(self, state: Tensor, card: Card|None, mask: Tensor|None = None) -> float|None:
        """ 存入蓄水池並觸發訓練 """
        self.steps_since_update += 1
        self.memory.push(state, card, mask)
        # 達標且蓄水池資料足夠時觸發
        if (self.steps_since_update >= self.update_every and self.memory.size() >= self.warmup):
            self.steps_since_update = 0
            return self.update()
        return None

    def update(self) -> float:
        """ 執行一次 Policy Network 的監督學習更新 """
        states, actions, masks = to_device(*self.memory.sample(self.batch_size))  # type: ignore
        # 只在合法動作上做 softmax (被選的動作必定合法)
        logits: Tensor = self.p_net(states).masked_fill(~masks, float("-inf"))  # shape: (batch_size, 61)
        loss: Tensor = F.cross_entropy(logits, actions)
        # 反向傳播與優化更新
        self.optimizer.zero_grad()
        loss.backward()       # pyright: ignore[reportUnknownMemberType]
        self.optimizer.step() # pyright: ignore[reportUnknownMemberType]
        return float(loss.item())

    def save(self) -> None:
        """ 儲存的訓練資料 """
        self.memory.save()
        self.p_net.save()

    def load(self) -> None:
        """ 讀取權重檔 """
        self.memory.load()
        self.p_net.load()
