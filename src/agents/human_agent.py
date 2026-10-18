from core import Card, Value, Color
from typing import TYPE_CHECKING
from .agent import Agent

if TYPE_CHECKING: from game import UnoGame

class HumanAgent(Agent):
    def __init__(self, eta: float = 0.85) -> None:
        super().__init__(eta)
    
    def set_game(self, game: 'UnoGame') -> None:
        super().set_game(game)

    """ 人類操控的Agent """
    def gen_prob_vector(self) -> tuple[float, ...]:
        """ 讓人類輸入要打的牌 """
        text: str = input("請輸入要打的牌(如: R|+4): ").strip()
        if (text == "draw"): id_: int = 60
        else:
            color_str, value_str = (text.split("|"))
            color: Color = Color.from_str(color_str)
            value: Value = Value.from_str(value_str)
            id_ = Card(color, value).id
        return tuple((1 if i==id_ else 0) for i in range(61))
