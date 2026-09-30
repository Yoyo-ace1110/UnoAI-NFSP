from abc import ABC, abstractmethod
# from core import Card

class Agent(ABC): 
    def gen_prob_vector(self) -> tuple[float, ...]:
        """ 輸出出各種牌的機率
            回傳一個出牌機率向量
            索引值對應到牌的種類
            最後一個元素代表抽牌 
        """
        ...
