from abc import ABC, abstractmethod
from core import Card

class UnoGame: ...

class Agent(ABC):
    # 共用的全域資訊
    game_ref: UnoGame
    
    @abstractmethod
    def gen_prob_vector(self) -> tuple[float, ...]:
        """ 輸出出各種牌的機率
            回傳一個出牌機率向量
            索引值對應到牌的種類
            最後一個元素代表抽牌 
        """
        ...
    
    @abstractmethod
    def play_drawed(self, card: Card) -> bool:
        """ 判斷是否打掉這張摸來的牌 """
        ...
