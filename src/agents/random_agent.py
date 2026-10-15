from random import uniform
from .agent import Agent

class RandomAgent(Agent):
    def gen_prob_vector(self) -> tuple[float, ...]:
        """ 回傳一個純隨機的機率向量 """
        return tuple(uniform(0, 1) for _ in  range(61))
