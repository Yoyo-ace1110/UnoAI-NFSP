import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .device import DEVICE, to_device
from torch import Tensor
from core import Card

NUM_ACTIONS: int = 61   # 60 種牌 + 抽牌

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, file_path: str = "q_network_v1.bin") -> None:
        """ Quality Network """
        super().__init__()
        self.file_path: str = file_path
        self.fc1: nn.Linear = nn.Linear(state_dim, 256) # 全連接層  輸入=>256
        self.fc2: nn.Linear = nn.Linear(256, 256)       # 隱藏層    256=>256
        self.fc3: nn.Linear = nn.Linear(256, 256)       # 隱藏層    256=>256
        self.out: nn.Linear = nn.Linear(256, NUM_ACTIONS) # 輸出層  256=>輸出
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
        print(f"Q-Network 權重已成功儲存至: {self.file_path}")
    
    def load(self) -> bool:
        """ 讀取權重檔，成功回傳 True。結構不符(如舊版狀態向量)或檔案不存在回傳 False，不丟例外 """
        if (not os.path.exists(self.file_path)): 
            print(f"找不到權重檔 {self.file_path}。")
            return False
        try:
            self.load_state_dict(torch.load(self.file_path, map_location=DEVICE, weights_only=True))
        except Exception as error:
            print(f"無法載入 {self.file_path} (可能是舊版狀態維度的權重，請刪除或重訓): {str(error).splitlines()[0]}")
            return False
        self.to(DEVICE)
        print(f"已成功載入 Q-Network 權重: {self.file_path}")
        return True

class ReplayBuffer:
    def __init__(self, file_path: str = "ReplayBuffer_v1.bin", capacity: int = 100000) -> None:
        """ Q-Network 記憶庫 (環形緩衝區；預先配置的大張量，取樣時不需要 torch.stack 一堆小張量) """
        self.file_path: str = file_path
        self.capacity: int = capacity
        self._size: int = 0
        self._pos: int = 0
        self._states:      Tensor|None = None
        self._actions:     Tensor|None = None
        self._rewards:     Tensor|None = None
        self._next_states: Tensor|None = None
        self._dones:       Tensor|None = None
        self._next_masks:  Tensor|None = None
        self._masks:       Tensor|None = None

    def _alloc(self, state_dim: int) -> None:
        c: int = self.capacity
        self._states      = torch.zeros((c, state_dim), dtype=torch.float32)
        self._next_states = torch.zeros((c, state_dim), dtype=torch.float32)
        self._actions     = torch.zeros(c, dtype=torch.int64)
        self._rewards     = torch.zeros(c, dtype=torch.float32)
        self._dones       = torch.zeros(c, dtype=torch.float32)
        self._next_masks  = torch.ones((c, NUM_ACTIONS), dtype=torch.bool)
        self._masks       = torch.ones((c, NUM_ACTIONS), dtype=torch.bool)

    def size(self) -> int: return self._size

    def push(
        self, 
        state: Tensor, 
        card: Card|None, 
        reward: float, 
        next_state: Tensor, 
        done: bool, 
        next_mask: Tensor|None = None,
        mask: Tensor|None = None
    ) -> None:
        """ 將一步遊戲經驗存入記憶庫 (FIFO)。next_mask 是「下一個決策點」的合法動作遮罩，mask 是「這個狀態」的 """
        if (self._states is None): self._alloc(state.shape[0])
        assert (self._actions is not None and self._rewards is not None and self._dones is not None)
        assert (self._next_states is not None and self._next_masks is not None and self._masks is not None)
        i: int = self._pos
        self._states[i]      = state
        self._actions[i]     = (60 if card is None else card.id)
        self._rewards[i]     = reward
        self._next_states[i] = next_state
        self._dones[i]       = float(done)
        self._next_masks[i]  = (True if next_mask is None else next_mask)
        self._masks[i]       = (True if mask is None else mask)
        self._pos  = (i + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[Tensor, ...]:
        """ 隨機抽樣一個批次進行訓練: (states, actions, rewards, next_states, dones, next_masks) """
        assert (self._states is not None and self._actions is not None and self._rewards is not None)
        assert (self._next_states is not None and self._dones is not None and self._next_masks is not None)
        assert (self._masks is not None)
        idx: Tensor = torch.randint(self._size, (batch_size,))
        return (
            self._states[idx], self._actions[idx], self._rewards[idx], 
            self._next_states[idx], self._dones[idx], self._next_masks[idx], self._masks[idx]
        )

    def save(self) -> None:
        """ 儲存二進位訓練資料 """ 
        if (self._states is None or self._size == 0):
            print(f"ReplayBuffer 是空的，略過儲存: {self.file_path}")
            return
        assert (self._actions is not None and self._rewards is not None and self._dones is not None)
        assert (self._next_states is not None and self._next_masks is not None and self._masks is not None)
        n: int = self._size
        torch.save({
            "states": self._states[:n], "actions": self._actions[:n], "rewards": self._rewards[:n],
            "next_states": self._next_states[:n], "dones": self._dones[:n], "next_masks": self._next_masks[:n],
            "masks": self._masks[:n],
        }, self.file_path)
        print(f"ReplayBuffer 已成功儲存至: {self.file_path} ({n} 筆)")

    def load(self) -> bool:
        """ 讀取二進位訓練資料 (格式不符的舊檔會被略過) """
        if (not os.path.exists(self.file_path)): 
            print(f"無法從 {self.file_path} 讀取 ReplayBuffer ")
            return False
        try:
            data: dict[str, Tensor] = torch.load(self.file_path, map_location="cpu", weights_only=True)
            n: int = min(int(data["states"].shape[0]), self.capacity)
            self._alloc(int(data["states"].shape[1]))
            assert (self._states is not None and self._actions is not None and self._rewards is not None)
            assert (self._next_states is not None and self._dones is not None and self._next_masks is not None and self._masks is not None)
            self._states[:n]      = data["states"][-n:]
            self._actions[:n]     = data["actions"][-n:]
            self._rewards[:n]     = data["rewards"][-n:]
            self._next_states[:n] = data["next_states"][-n:]
            self._dones[:n]       = data["dones"][-n:]
            self._next_masks[:n]  = data["next_masks"][-n:]
            if ("masks" in data): self._masks[:n] = data["masks"][-n:]   # type: ignore
        except Exception as error:
            print(f"無法讀取 {self.file_path} (舊版格式?)，略過: {str(error).splitlines()[0]}")
            self._states = None
            self._size = self._pos = 0
            return False
        self._size = n
        self._pos  = n % self.capacity
        print(f"已成功從 {self.file_path} 讀取 ReplayBuffer ({n} 筆)")
        return True

class DQNTrainer:
    def __init__(
        self, 
        state_dim: int, 
        lr: float = 1e-3, 
        gamma: float = 0.99, 
        batch_size: int = 64, 
        network_path: str = "q_network_v1.bin", 
        memory_path: str = "ReplayBuffer_v1.bin",
        update_every: int = 8,
        warmup: int = 1000,
        q_clip: float = 3.5,
        cql_alpha: float = 0.0,
    ) -> None:
        """ Deep Q-Network Trainer
        cql_alpha: 保守 Q 學習 (CQL) 的強度。loss 額外加上 α*(logsumexp_合法動作 Q − 實際動作 Q)，
                   把「資料裡沒做過的動作」(如示範者從不做的自發抽牌) 的 Q 往下壓，避免 max 引用它們的外推值而高估。0=關閉
        q_clip: 目標 Q 值截斷範圍。終局 ±1 加上每步 ±0.025 的 shaping，折扣後回報不會超過約 ±3.5，
                超出代表外推誤差在滾雪球，截斷可避免發散 (只是安全網，不能取代正確的資料)
        update_every: 每新增幾筆資料做一次梯度更新
        warmup: 記憶庫至少要有幾筆才開始訓練
        """
        self.gamma: float = gamma
        self.q_clip: float = q_clip
        self.cql_alpha: float = cql_alpha
        self.batch_size: int = batch_size
        self.update_every: int = update_every
        self.warmup: int = max(warmup, batch_size)
        self.memory: ReplayBuffer = ReplayBuffer(file_path=memory_path)
        self.q_net: QNetwork = QNetwork(state_dim, file_path=network_path)
        # 設定目標網路避免球員兼裁判
        self.target_net: QNetwork = QNetwork(state_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer: optim.Adam = optim.Adam(self.q_net.parameters(), lr)
        # 用來計算增加多少筆資料的計數器
        self.target_update_count: int = 0
        self.steps_since_update: int = 0
        self.target_net.eval()

    def push(
        self, 
        state: Tensor, 
        card: Card|None, 
        reward: float, 
        next_state: Tensor, 
        done: bool, 
        next_mask: Tensor|None = None,
        mask: Tensor|None = None
    ) -> float|None:
        """ 存入記憶庫，並檢查是否該觸發 update。有更新時回傳 loss """
        self.steps_since_update += 1
        self.memory.push(state, card, reward, next_state, done, next_mask, mask)
        if (self.steps_since_update >= self.update_every and self.memory.size() >= self.warmup):
            self.steps_since_update = 0
            return self.update()
        return None

    def update(self) -> float:
        """ 執行一次 DQN 梯度下降更新 """
        states, actions, rewards, next_states, dones, next_masks, masks = to_device(*self.memory.sample(self.batch_size))
        # 計算當前狀態的 Q 值
        all_q: Tensor = self.q_net(states)
        q_values: Tensor = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        # 計算目標 Q 值 (Bellman Equation)
        with torch.no_grad():
            # 只在「下一個決策點的合法動作」中取最大 Q 值，非法動作的 Q 從未被訓練，不能參與 max
            next_q: Tensor = self.target_net(next_states).masked_fill(~next_masks, float("-inf")).max(1).values
            # 終局沒有未來回報 (用 where 避免 -inf * 0 = nan)
            next_q = torch.where(dones > 0.5, torch.zeros_like(next_q), next_q)
            target_q_values: Tensor = (rewards + self.gamma*next_q).clamp(-self.q_clip, self.q_clip)
        # 計算 Loss (Huber 對離群值比 MSE 穩定)
        loss: Tensor = F.smooth_l1_loss(q_values, target_q_values)
        # CQL: 壓低合法動作中「資料沒做過」的 Q (logsumexp 是 max 的平滑版，實際動作的 Q 不受影響)
        if (self.cql_alpha > 0):
            cql: Tensor = (torch.logsumexp(all_q.masked_fill(~masks, float("-inf")), dim=1) - q_values).mean()
            loss = loss + self.cql_alpha*cql
        # 反向傳播與優化更新
        self.optimizer.zero_grad()
        loss.backward()       # pyright: ignore[reportUnknownMemberType]
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.optimizer.step() # pyright: ignore[reportUnknownMemberType]
        # 定期同步 target network
        self.target_update_count += 1
        if (self.target_update_count % 100 == 0):
            self.target_net.load_state_dict(self.q_net.state_dict())
        return float(loss.item())

    def save(self) -> None:
        """ 儲存的訓練資料 """
        self.memory.save()
        self.q_net.save()

    def load(self) -> None:
        """ 讀取權重檔 """
        self.q_net.load()
        self.memory.load()
        self.target_net.load_state_dict(self.q_net.state_dict())
