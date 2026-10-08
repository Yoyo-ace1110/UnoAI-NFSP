from random import randint, uniform, random
from core import Card, Color
from .agent import Agent

class RandomAgent(Agent):
    def gen_prob_vector(self) -> tuple[float, ...]:
        """ 回傳一個純隨機的機率向量 """
        return tuple(uniform(0, 1) for _ in  range(61))
    
    def play_drawed(self, card: Card) -> bool:
        """ 回傳一個純隨機 [True, False] """
        return bool(random() < 0.5)

    def decide_color_for_black(self) -> Color:
        """ 回傳一個純隨機的顏色 """
        return Color(randint(0, 3))
