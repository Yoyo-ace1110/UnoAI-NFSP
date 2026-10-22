from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from torch import Tensor
from core import Card
import weakref

if TYPE_CHECKING: from engine import UnoGame

class Agent(ABC):
    def __init__(
        self, 
        eta: float = 0.85, 
        p_network_path: str|None = None, 
        q_network_path: str|None = None, 
        replay_buffer_path: str|None = None, 
        reservior_memory_path: str|None = None, 
    ) -> None:
        """ 設定 eta 以初始化 Agent """
        self.eta: float = eta
        self.q_net_path   : str|None = q_network_path
        self.p_net_path   : str|None = p_network_path
        self.q_memory_path: str|None = replay_buffer_path
        self.p_memory_path: str|None = reservior_memory_path

    def save(self) -> None: pass
    
    def eval(self) -> None: pass
    
    def train(self) -> None: pass
    
    def push_p(self, state: Tensor, card: Card|None) -> float|None: pass

    def push_q(self, state: Tensor, card: Card|None, reward: float, next_state: Tensor, done: bool) -> float|None: pass
    
    def reload(self, batch_size: int = 64, gamma: float = 0.99, lr: float = 0.01) -> None: pass
    
    def set_game(self, game: 'UnoGame') -> None:
        """ get weak reference of game """
        if hasattr(self, 'game'): return
        self.game: 'UnoGame' = weakref.proxy(game)
        self.reload()
        self.eval()
    
    def get_tensor(self) -> Tensor:
        """ 取得遊戲狀態張量 """
        return self.game.get_tensor()
    
    @abstractmethod
    def gen_prob_vector(self) -> tuple[float, ...]:
        """ 輸出出各種牌的機率
            回傳一個出牌機率向量
            索引值對應到牌的種類
            最後一個元素代表抽牌 
        """
        ...
