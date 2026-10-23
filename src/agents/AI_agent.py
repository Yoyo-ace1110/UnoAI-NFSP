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
    ) -> None:
        """ AI Agent --version 1 """
        super().__init__(
            eta, 
            p_network_path, 
            q_network_path, 
            replay_buffer_path, 
            reservior_memory_path
        )
        self.no_grad: bool = True

    @property
    def q_net(self) -> QNetwork: return self.q_trainer.q_net
    
    @property
    def p_net(self) -> PNetwork: return self.p_trainer.p_net

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

    def reload(self, batch_size: int = 64, gamma: float = 0.99, lr: float = 0.01) -> None:
        """ 重新載入模型 """
        state_dim: int = self.get_tensor().shape[0]
        if (self.q_net_path is None): self.q_net_path = "q_network_v1.bin"
        if (self.p_net_path is None): self.p_net_path = "p_network_v1.bin"
        if (self.q_memory_path is None): self.q_memory_path = "ReplayBuffer_v1.bin"
        if (self.p_memory_path is None): self.p_memory_path = "ReservoirMemory_v1.bin"
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
        """ 根據 eta 切換模型並預測 """
        if (self.no_grad):
            with torch.no_grad(): 
                return (self._policy() if (random() > self.eta) else self._quality())
        return (self._policy() if (random() > self.eta) else self._quality())

    def _quality(self) -> tuple[float, ...]:
        """ predict by Q-Network """
        state: Tensor = to_device(self.get_tensor()).unsqueeze(0)
        q_values: list[float] = self.q_net(state).squeeze(0).cpu().tolist()
        # 合法化 q_values
        for i in range(len(q_values)):
            # from_id 要跳過抽牌
            if (i == 60): continue
            if (not self.can_play(Card.from_id(i))):
                # Q值可以是任意實數，所以設為最小
                q_values[i] = float('-inf')
        # 只留下 Q 值最大者
        max_index: int = max(range(len(q_values)), key=lambda i: q_values[i])
        return tuple(float(i==max_index) for i in range(len(q_values)))
    
    def _policy(self) -> tuple[float, ...]:
        """ predict by Policy Network """
        state: Tensor = to_device(self.get_tensor()).unsqueeze(0)
        prob_tensor: Tensor = torch.softmax(self.p_net(state), dim=-1)
        prob_vec: list[float] = prob_tensor.squeeze(0).cpu().tolist() # type: ignore
        # 合法化 prob_vec
        for i in range(len(prob_vec)):
            # from_id 要跳過抽牌
            if (i == 60): continue
            if (not self.can_play(Card.from_id(i))):
                prob_vec[i] = 0.0 # 不合法者機率為零
        return tuple(self.softmax(prob_vec))
