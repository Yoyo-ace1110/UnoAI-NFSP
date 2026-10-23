from models import DQNTrainer, PolicyTrainer, QNetwork, PNetwork, to_device
from random import random
from .agent import Agent
from torch import Tensor
from core import Card
import torch

class AIAgent(Agent):
    def __init__(
        self, 
        eta: float = 0.85, 
        p_network_path: str|None = None, 
        q_network_path: str|None = None, 
        replay_buffer_path: str|None = None, 
        reservior_memory_path: str|None = None, 
        inference_only: bool = False
    ) -> None:
        """ AI Agent --version 1 """
        super().__init__(
            eta, 
            p_network_path, 
            q_network_path, 
            replay_buffer_path, 
            reservior_memory_path, 
            inference_only
        )
        self._q_net: QNetwork|None = None
        self._p_net: PNetwork|None = None

    @property
    def q_net(self) -> QNetwork: return (self._q_net if self.inference_only else self.q_trainer.q_net) # type: ignore
    
    @property
    def p_net(self) -> PNetwork: return (self._p_net if self.inference_only else self.p_trainer.p_net) # type: ignore
    
    def eval(self) -> None:
        """ 評估模式 """
        self.q_net.eval()
        self.p_net.eval()
        self.no_grad = True

    def train(self) -> None:
        """ 訓練模式 """
        self.q_net.train()
        self.p_net.train()
        self.no_grad = False

    def reload(self, batch_size: int = 64, gamma: float = 0.99, lr: float = 0.001) -> None:
        """ 重新載入模型 """
        state_dim: int = self.get_tensor().shape[0]
        if (self.q_net_path is None): self.q_net_path = "q_network_v1.bin"
        if (self.p_net_path is None): self.p_net_path = "p_network_v1.bin"
        if (self.q_memory_path is None): self.q_memory_path = "ReplayBuffer_v1.bin"
        if (self.p_memory_path is None): self.p_memory_path = "ReservoirMemory_v1.bin"
        # 僅載入神經網路
        if self.inference_only:
            self._q_net = QNetwork(state_dim, self.q_net_path)
            self._q_net.load()
            self._p_net = PNetwork(state_dim, self.p_net_path)
            self._p_net.load()
            return
        # 也載入記憶庫
        self.p_trainer: PolicyTrainer = PolicyTrainer(state_dim, lr,  batch_size, self.p_net_path, self.p_memory_path)
        self.q_trainer: DQNTrainer = DQNTrainer(state_dim, lr, gamma, batch_size, self.q_net_path, self.q_memory_path)
        self.q_trainer.load()
        self.p_trainer.load()

    def push_p(self, state: Tensor, card: Card|None) -> float|None: 
        return self.p_trainer.push(state, card)

    def push_q(self, state: Tensor, card: Card|None, reward: float, next_state: Tensor, done: bool) -> float|None: 
        return self.q_trainer.push(state, card, reward, next_state, done)

    def save(self) -> None:
        self.q_trainer.save()
        self.p_trainer.save()
    
    def get_prob_vector(self) -> tuple[float, ...]:
        """ 根據 eta 選擇網路進行預測 """
        return (self._policy() if (random() > self.eta) else self._quality())

    def _quality(self) -> tuple[float, ...]:
        """ predict by Q-Network with inference_mode and legal_ids """
        with torch.inference_mode():
            state: Tensor = to_device(self.get_tensor()).unsqueeze(0)
            q_values: Tensor = self.q_net(state).squeeze(0)
            legal: list[int] = self.legal_ids()
            # 建立遮罩
            masked_q: Tensor = torch.full((61,), float('-inf'), device=q_values.device)
            # 將合法動作的 Q 值填入遮罩
            for idx in legal: masked_q[idx] = q_values[idx]
            masked_q[60] = q_values[60]
            # 選出 Q 值最大者
            max_index: int = int(torch.argmax(masked_q).item())
            return tuple(float(i == max_index) for i in range(61))
    
    def _policy(self) -> tuple[float, ...]:
        """ predict by Policy Network with inference_mode and legal_ids """
        with torch.inference_mode():
            state: Tensor = to_device(self.get_tensor()).unsqueeze(0)
            logits: Tensor = self.p_net(state).squeeze(0)
            legal: list[int] = self.legal_ids()
            masked_logits: Tensor = torch.full((61,), -1e9, device=logits.device)
            # 將合法動作的 Logits 填入遮罩
            for idx in legal: masked_logits[idx] = logits[idx]
            masked_logits[60] = logits[60]
            # 透過 Masked Softmax 計算機率分佈
            prob_tensor: Tensor = torch.softmax(masked_logits, dim=-1)
            return tuple(prob_tensor.cpu().tolist()) # type: ignore
