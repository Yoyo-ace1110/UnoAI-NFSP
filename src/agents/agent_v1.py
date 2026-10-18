from models import QNetwork, PNetwork, to_device
from typing import TYPE_CHECKING
from random import random
from .agent import Agent
from torch import Tensor
import torch

if TYPE_CHECKING: from game import UnoGame

class Agent_v1(Agent):
    def __init__(self, eta: float = 0.85) -> None:
        """ AI Agent --version 1 """
        super().__init__(eta=eta)

    def reload(self) -> None:
        """ 重新載入模型 """
        state_dim: int = self.get_tensor().shape[0]
        self.q_net: QNetwork = QNetwork(state_dim)
        self.p_net: PNetwork = PNetwork(state_dim)
        self.q_net.load()
        self.p_net.load()
        self.q_net.eval()
        self.p_net.eval()

    def set_game(self, game: 'UnoGame') -> None:
        super().set_game(game)
    
    def gen_prob_vector(self) -> tuple[float, ...]:
        """ 根據 eta 切換模型並預測 """
        if (random() > self.eta):
            return self._policy()
        return self._quality()

    def _quality(self) -> tuple[float, ...]:
        """ predict by Q-Network """
        with torch.no_grad():
            state: Tensor = to_device(self.get_tensor()).unsqueeze(0)
            q_values: Tensor = self.q_net(state)
            return tuple(q_values.squeeze(0).cpu().tolist()) # type: ignore
        
    def _policy(self) -> tuple[float, ...]:
        """ predict by Policy Network """
        with torch.no_grad():
            state: Tensor = to_device(self.get_tensor()).unsqueeze(0)
            probabilities: Tensor = torch.softmax(self.p_net(state), dim=-1)
            return tuple(probabilities.squeeze(0).cpu().tolist()) # type: ignore
