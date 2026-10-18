from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from torch import Tensor
import weakref

if TYPE_CHECKING: from game import UnoGame

class Agent(ABC):
    def __init__(self, eta: float = 0.85) -> None:
        """ 設定 eta 以初始化 Agent """
        self.eta: float = eta
    
    def reload(self) -> None: pass
    
    def set_game(self, game: 'UnoGame') -> None:
        """ get weak reference of game """
        self.game: 'UnoGame' = weakref.proxy(game)
        self.reload()
    
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
