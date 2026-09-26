from core import Card

class Player:
    def __init__(self, cards: list[Card]|None) -> None:
        """ 初始化玩家 """
        self.cards: list[Card] = (cards if (cards is not None) else [])

    def set_hand(self, cards: list[Card]) -> None:
        """ 賦予手牌 """
        self.cards = cards
