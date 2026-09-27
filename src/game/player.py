from core import Card, Deck, Rule
from agents import Agent

class Player:
    def __init__(self, agent: Agent) -> None:
        """ 初始化玩家 """
        self.agent: Agent = agent
        self.hand: list[Card] = []

    def deal_hand(self, cards: list[Card]) -> None:
        """ 賦予手牌 """
        self.hand = cards

    def show_hand(self, index: int) -> None:
        """ 輸出手牌 """
        print(f"Player{index}: ", end="")
        for card in self.hand: 
            print(f"{card} ", end="")
        print() # 最後的換行

    def hand_size(self) -> int: return len(self.hand)

    def is_winner(self) -> bool:
        """ 判斷自己是否贏了 """
        return (self.hand_size() == 0)

    def uno(self) -> bool:
        """ 判斷 Uno 了沒 """
        return (self.hand_size() == 1)

    def draw(self, deck: Deck, n: int) -> None:
        """ 抽牌 """
        self.hand.extend(deck.draw(n))

    def receive(self, card: Card) -> None:
        """ 抽單一一張牌 """
        self.hand.append(card)

    def draw_to_play(self, deck: Deck) -> None:
        """ 抽牌直到有牌可以打, 並自動打出 """
        while (True):
            # 抽牌
            card: Card|None = deck.draw_one()
            # 沒牌抽了
            if (card is None): return None
            # 可以打
            if (deck.is_playable(card)): deck.set_top(card)
            # 不能打, 繼續抽
            self.hand.append(card)

    def genmove(self) -> Card|None:
        """ 出牌, None代表抽牌 """
        # TODO:
    
    def play_drawed(self, card: Card) -> bool:
        """ 判斷是否打掉這張抽來的牌 """
        # TODO:
        if (not Rule.play_after_draw): return False
