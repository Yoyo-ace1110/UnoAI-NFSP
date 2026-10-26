import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from .device import DEVICE, to_device
from torch import Tensor
from core import Card

# 60 種牌 + 抽牌
NUM_ACTIONS: int = 61 

class QNetwork(nn.Module):
    def __init__(self, state_dim: int, file_path: str = "q_network_v1.bin") -> None:
        """ Quality Network """
        super().__init__()
        self.file_path: str = file_path
        self.fc1: nn.Linear = nn.Linear(state_dim, 256)   # 全連接層  輸入=>256
        self.fc2: nn.Linear = nn.Linear(256, 256)         # 隱藏層    256=>256
        self.fc3: nn.Linear = nn.Linear(256, 256)         # 隱藏層    256=>256
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
        """ 讀取權重檔, 成功回傳 True。結構不符(如舊版狀態向量)或檔案不存在回傳 False, 不丟例外 """
        if (not os.path.exists(self.file_path)): 
            print(f"找不到權重檔 {self.file_path}。")
            return False
        try:
            self.load_state_dict(torch.load(self.file_path, map_location=DEVICE, weights_only=True))
        except Exception as error:
            print(f"無法載入 {self.file_path} (可能是舊版狀態維度的權重, 請刪除或重訓): {str(error).splitlines()[0]}")
            return False
        self.to(DEVICE)
        print(f"已成功載入 Q-Network 權重: {self.file_path}")
        return True

class ReplayBuffer:
    def __init__(self, file_path: str = "ReplayBuffer_v1.bin", capacity: int = 64000) -> None:
        """ Q-Network 記憶庫"""
        self._pos: int = 0
        self._size: int = 0
        self._protect: int = 0   # 前 _protect 筆資料不會被覆蓋 (示範資料)
        self.capacity: int = capacity
        self.file_path: str = file_path
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

    def freeze(self, max_ratio: float = 0.3) -> None:
        """ 把目前已有的資料 (示範資料) 保護起來, 之後環形緩衝區只覆蓋其後的資料。
            否則記憶庫寫滿後, 最舊的示範資料會被自己產生的資料依序蓋掉, CQL 就失去示範者行為這個錨。
            max_ratio: 受保護的資料最多占容量的比例, 避免新資料沒地方寫
        """
        self._protect = min(self._size, int(self.capacity*max_ratio))

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
        """ 將一步遊戲經驗存入記憶庫 (FIFO, 受保護的前 _protect 筆不會被覆蓋) """
        if (self._states is None): self._alloc(state.shape[0])
        assert (self._states is not None)
        assert (self._actions is not None)
        assert (self._rewards is not None)
        assert (self._next_states is not None)
        assert (self._dones is not None)
        assert (self._next_masks is not None)
        assert (self._masks is not None)
        i: int = self._pos
        self._states[i]      = state
        self._actions[i]     = (60 if card is None else card.id)
        self._rewards[i]     = reward
        self._next_states[i] = next_state
        self._dones[i]       = float(done)
        self._next_masks[i]  = (True if next_mask is None else next_mask)
        self._masks[i]       = (True if mask is None else mask)
        self._pos  = (i + 1) if (i + 1 < self.capacity) else self._protect
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple[Tensor, ...]:
        """ 隨機抽樣一個批次進行訓練: (states, actions, rewards, next_states, dones, next_masks) """
        assert (self._states is not None)
        assert (self._actions is not None)
        assert (self._rewards is not None)
        assert (self._next_states is not None)
        assert (self._dones is not None)
        assert (self._next_masks is not None)
        assert (self._masks is not None)
        index: Tensor = torch.randint(self._size, (batch_size,))
        return (
            self._states[index], 
            self._actions[index], 
            self._rewards[index], 
            self._next_states[index], 
            self._dones[index], 
            self._next_masks[index], 
            self._masks[index]
        )

    def save(self) -> None:
        """ 儲存二進位訓練資料 """ 
        if ((self._states is None) or (self._size == 0)):
            print(f"ReplayBuffer 是空的，略過儲存: {self.file_path}")
            return
        assert (self._states is not None)
        assert (self._actions is not None)
        assert (self._rewards is not None)
        assert (self._next_states is not None)
        assert (self._dones is not None)
        assert (self._next_masks is not None)
        assert (self._masks is not None)
        n: int = self._size
        torch.save(
            {
                "states": self._states[:n], 
                "actions": self._actions[:n], 
                "rewards": self._rewards[:n],
                "next_states": self._next_states[:n], 
                "dones": self._dones[:n], 
                "next_masks": self._next_masks[:n],
                "masks": self._masks[:n],
                "protect": self._protect,
            }, 
            self.file_path
        )
        print(f"ReplayBuffer 已成功儲存至: {self.file_path} ({n} 筆)")

    def load(self) -> bool:
        """ 讀取二進位訓練資料 """
        if (not os.path.exists(self.file_path)): 
            print(f"無法從 {self.file_path} 讀取 ReplayBuffer ")
            return False
        try:
            data = torch.load(self.file_path, map_location="cpu", weights_only=True)  # type: ignore
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
            self._size = 0
            self._pos = 0
            self._protect = 0
            return False
        self._size = n
        self._protect = (min(int(data["protect"]), n) if ("protect" in data) else 0)  # type: ignore
        self._pos  = (n if (n < self.capacity) else self._protect)
        print(f"已成功從 {self.file_path} 讀取 ReplayBuffer ({n} 筆, 其中 {self._protect} 筆受保護)")
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
        cql_alpha: float = 0.2,
        double_dqn: bool = True,
    ) -> None:
        """ Deep Q-Network Trainer
            cql_alpha: 
                保守 Q 學習 (CQL) 的強度。loss 額外加上 alpha*(logsumexp_合法動作 Q - 實際動作 Q), 
                把資料裡沒有的動作(如示範者從不主動抽牌) 的 Q 往下壓, 避免 max 引用它們的外推值而高估。
                測試值 0.1~0.3 (rule=True/False 的離線測試中, 0.1~0.3 都穩定, 0.03 太弱, 0 會發散)
            double_dqn: 
                用線上網路選動作、目標網路評價。rule=True 的 selfplay(eta=0.85) 不加時約 1200 局發散(抽牌率 95%、全部截斷),
                加上後 2 個 seed 在 1800/2400 局都穩定 (縮小規模的重現)。
            q_clip: 目標 Q 值截斷範圍。終局 ±1 加上每步 ±0.025 的 shaping, 折扣後回報不會超過約 ±3.5, 
                超出代表外推誤差在滾雪球, 截斷可避免發散 (只是安全機制)
            update_every: 每新增幾筆資料做一次梯度更新
            warmup: 記憶庫至少要有幾筆才開始訓練
        """
        self.gamma: float = gamma
        self.q_clip: float = q_clip
        self.cql_alpha: float = cql_alpha
        self.double_dqn: bool = double_dqn   # True: Double DQN (見 update)
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
        """ 存入記憶庫, 並檢查是否該觸發 update。有更新時回傳 loss """
        self.steps_since_update += 1
        self.memory.push(state, card, reward, next_state, done, next_mask, mask)
        if (self.steps_since_update >= self.update_every and self.memory.size() >= self.warmup):
            self.steps_since_update = 0
            return self.update()
        return None

    def update(self) -> float:
        """ 執行一次 DQN 梯度下降更新 """
        samples: tuple[Tensor, ...] = self.memory.sample(self.batch_size)
        states, actions, rewards, next_states, dones, next_masks, masks = to_device(*samples)  # type: ignore
        # 計算當前狀態的 Q 值
        all_q: Tensor = self.q_net(states)
        q_values: Tensor = all_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        # 計算目標 Q 值 (Bellman Equation)
        with torch.no_grad():
            # 只在「下一個決策點的合法動作」中取最大 Q 值, 非法動作的 Q 從未被訓練, 不參與 max
            target_all: Tensor = self.target_net(next_states).masked_fill(~next_masks, float("-inf"))
            if (self.double_dqn):
                # Double DQN: 用「線上網路」選動作、「目標網路」評價，避免 max 操作把估計噪音系統性地往上選 (過度樂觀)
                best_actions: Tensor = self.q_net(next_states).masked_fill(~next_masks, float("-inf")).argmax(1, keepdim=True)
                next_q: Tensor = target_all.gather(1, best_actions).squeeze(1)
            else:
                next_q = target_all.max(1).values
            # 終局沒有未來回報 (用 where 避免 -inf * 0 = nan)
            next_q = torch.where(dones > 0.5, torch.zeros_like(next_q), next_q)
            target_q_values: Tensor = (rewards + self.gamma*next_q).clamp(-self.q_clip, self.q_clip)
        # 計算 Loss (Huber 對離群值比 MSE 穩定)
        loss: Tensor = F.smooth_l1_loss(q_values, target_q_values)
        # CQL: 壓低合法動作中資料沒做過的 Q 
        # logsumexp 是 max 的平滑版，實際動作的 Q 不受影響
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
