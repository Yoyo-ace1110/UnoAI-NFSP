from models import DQNTrainer, PolicyTrainer, QNetwork, PNetwork, to_device
from typing import TYPE_CHECKING
from random import random
from .agent import Agent
from torch import Tensor
from core import Card
import torch

if TYPE_CHECKING: from game import UnoGame

class Agent_v1(Agent):
    def __init__(self, eta: float = 0.85) -> None:
        """ AI Agent --version 1 """
        super().__init__(eta=eta)
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

    def set_game(self, game: 'UnoGame') -> None:
        super().set_game(game)
    
    def reload(self, batch_size: int = 64, gamma: float = 0.99, lr: float = 0.001) -> None:
        """ 重新載入模型 """
        state_dim: int = self.get_tensor().shape[0]
        self.q_trainer: DQNTrainer = DQNTrainer(state_dim, batch_size, gamma, lr)
        self.p_trainer: PolicyTrainer = PolicyTrainer(state_dim, batch_size, lr)
        self.q_trainer.load()
        self.p_trainer.load()
    
    def push_p(self, state: Tensor, card: Card|None) -> float|None: 
        return self.p_trainer.push(state, card)

    def push_q(self, state: Tensor, card: Card|None, reward: float, next_state: Tensor, done: bool) -> float|None: 
        return self.q_trainer.push(state, card, reward, next_state, done)

    def save(self) -> None:
        self.q_trainer.save()
        self.p_trainer.save()

    def gen_prob_vector(self) -> tuple[float, ...]:
        """ 根據 eta 切換模型並預測 """
        if (self.no_grad):
            with torch.no_grad(): 
                return (self._policy() if (random() > self.eta) else self._quality())
        return (self._policy() if (random() > self.eta) else self._quality())

    def _quality(self) -> tuple[float, ...]:
        """ predict by Q-Network """
        state: Tensor = to_device(self.get_tensor()).unsqueeze(0)
        q_values: Tensor = self.q_net(state)
        return tuple(q_values.squeeze(0).cpu().tolist()) # type: ignore
        
    def _policy(self) -> tuple[float, ...]:
        """ predict by Policy Network """
        with torch.no_grad():
            state: Tensor = to_device(self.get_tensor()).unsqueeze(0)
            probabilities: Tensor = torch.softmax(self.p_net(state), dim=-1)
            return tuple(probabilities.squeeze(0).cpu().tolist()) # type: ignore
