from typing import TYPE_CHECKING
from random import uniform
from .agent import Agent

if TYPE_CHECKING: from game import UnoGame

class RandomAgent(Agent):
    def __init__(self, eta: float = 0.85) -> None:
        super().__init__(eta)
    
    def set_game(self, game: 'UnoGame') -> None:
        super().set_game(game)
        
    def gen_prob_vector(self) -> tuple[float, ...]:
        """ 回傳一個純隨機的機率向量 """
        return tuple(uniform(0, 1) for _ in  range(61))
