from abc import ABC, abstractmethod
from core import Card

class Agent(ABC): 
    @abstractmethod
    def genmove(self) -> Card|None:
        """ 出牌, None代表抽牌 """
