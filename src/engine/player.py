from agents import Agent, HumanAgent
from core import Card, Deck
from random import choices
import bisect

class Player:
    def __init__(self, agent: Agent) -> None:
        """ 初始化玩家 """
        self.index: int = -1
        self.agent: Agent = agent
        self.hand: list[Card] = []

    def has(self, target: Card) -> bool:
        """ 檢查玩家是否有這張手牌 """
        # 黑牌直接找牌面相同者
        if (target.is_black()):
            for card in self.hand:
                if (card.value == target.value): 
                    return True
            return False
        # 顏色牌則一一比較
        return (target in self.hand)

    def reset(self, deck: Deck) -> None:
        """ 重置玩家 """
        deck.cards.extend(self.hand)
        self.hand.clear()

    def set_index(self, index: int) -> None:
        """ 設定編號 """
        self.index = index

    def hand_size(self) -> int: return len(self.hand)

    def deal_hand(self, cards: list[Card]) -> None:
        """ 賦予手牌 """
        self.hand = cards
        self.sort_hand()

    def show_hand(self) -> None:
        """ 輸出手牌 """
        print(f"{'Human' if self.is_human() else 'Computer'}{self.index}: ", end="")
        for card in self.hand: card.cout(show_color_for_black=False, end=" ")
        print()

    def sort_hand(self) -> None:
        """ 排序手牌 """
        self.hand.sort()
    
    def is_winner(self) -> bool:
        """ 判斷自己是否贏了 """
        return (self.hand_size() == 0)

    def is_human(self) -> bool:
        """ 判斷是否為人類玩家 """
        return isinstance(self.agent, HumanAgent)

    def uno(self) -> bool:
        """ 判斷 Uno 了沒 """
        return (self.hand_size() == 1)

    def draw(self, deck: Deck, n: int) -> None:
        """ 抽牌 """
        self.hand.extend(deck.draw(n))
        self.sort_hand()

    def remove(self, card: Card) -> None:
        """ 移除這張手牌 """
        if (not self.has(card)): raise ValueError("Player::remove(card) does not exist")
        if (card.is_color()): return self.hand.remove(card)
        # 拿掉相同牌面的第一張黑牌
        for i in range(len(self.hand)):
            if (self.hand[i].value == card.value): 
                self.hand.pop(i)
                return
        raise ValueError("Player::remove(card) does not exist")

    def receive(self, card: Card) -> None:
        """ 抽一張牌 (插入並排序)"""
        bisect.insort(self.hand, card)

    def is_all_black(self) -> int:
        """ 判斷手上是否全為黑色牌 """
        return (self.hand[0].is_black())

    def genmove(self) -> Card|None:
        """ 出牌, None代表抽牌 """
        prob_vec: tuple[float, ...] = self.agent.get_prob_vector()
        # 根據機率隨機挑選一張
        chosen_index: int = choices(
            population=range(len(prob_vec)), 
            weights=prob_vec, 
            k=1
        )[0]
        # 轉成牌張並回傳
        if (chosen_index == 60): return None
        return Card.from_id(chosen_index)
