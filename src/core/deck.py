import random
from .card import Card, Color, Value

# 牌堆
class Deck:
    def __init__(self) -> None:
        """ 初始化牌堆 """
        color: Color
        value: Value
        self.cards: list[Card] = []
        self.played: list[Card] = []
        # 加入 4 張 0
        value = Value(0)
        for i in range(4):
            color = Color(i)
            self.cards.append(Card(color, value))
        # 加入 1~9 各兩張
        for i in range(1, 10):
            value = Value(i)
            for j in range(4):
                color = Color(j)
                self.cards.append(Card(color, value))
        # 翻出第一張牌
        choice: int = random.randint(0, self.size()-1)
        self.top: Card = self.cards[choice]
        self.cards.pop(choice)
        # 加入功能牌
        for i in range(10, 13):
            value = Value(i)
            for j in range(4):
                color = Color(j)
                self.cards.append(Card(color, value))
        # 加入黑色牌
        for i in range(13, 15):
            value = Value(i)
            for j in range(4):
                color = Color(j)
                self.cards.append(Card(color, value))

    def size(self) -> int:
        """ 牌堆剩餘幾張牌 """
        return len(self.cards)

    def empty(self) -> bool:
        """ 判斷牌堆是否為空 """
        return (self.size() == 0)
    
    def shuffle(self) -> None:
        """ 牌堆洗牌 """
        random.shuffle(self.cards)
    
    def reshuffle(self) -> None:
        self.cards.extend(self.played)
        self.played.clear()
        self.shuffle()
    
    def set_top(self, card: Card) -> None:
        """ 更新牌頂 """
        self.played.append(self.top)
        self.top = card
    
    def draw_one(self) -> Card|None:
        """ 抽出一張牌 """
        if (self.empty()): self.reshuffle()
        if (self.empty()): return None
        return self.cards.pop(0)

    def draw(self, n: int) -> list[Card]:
        """ 抽出 n 張牌 """
        result: list[Card] = []
        for _ in range(n):
            drawed: Card|None = self.draw_one()
            if (drawed is not None): result.append(drawed)
        return result

    def is_playable(self, card: Card) -> bool:
        """ 判斷牌能不能打 """
        return card.is_playable_after(self.top)
