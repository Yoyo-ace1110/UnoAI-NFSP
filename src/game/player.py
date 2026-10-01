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
        for card in self.hand: print(f"{card} ", end="")
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

    def color_count(self) -> int:
        """ 判斷手上是否不是只剩黑色牌 """
        count: int = 0
        for card in self.hand:
            if (not card.is_black()):
                count += 1
        return count

    def softmax_playables(self, deck_top: Card) -> list[float]:
        """ 輸出手上各牌出牌的機率
            回傳一個出牌機率向量
            索引值對應到手牌索引
            最後一個元素代表抽牌 
        """
        playable_probs: list[float] = []
        color_count: int = self.color_count()
        prob_vec: tuple[float, ...] = self.agent.gen_prob_vector()
        # 其他牌機率
        for card in self.hand:
            # 不能打就跳過
            if (not card.is_playable_after(deck_top)): continue
            if (not Rule.black_finisher and self.uno() and card.is_black()): continue
            if (Rule.color_retention and card.is_color() and color_count == 1): continue
            # 如果可以打則加入向量
            playable_probs.append(prob_vec[card.id])
        # 抽牌機率
        playable_probs.append(prob_vec[-1])
        return playable_probs

    def genmove(self, deck_top: Card) -> Card|None:
        """ 出牌, None代表抽牌 """
        prob_vec: list[float] = self.softmax_playables(deck_top)
        max_index: int = max(range(len(prob_vec)), key=prob_vec.__getitem__)
        # 最後一個元素代表抽牌
        if (max_index == self.hand_size()): return None
        # 其他則去手牌上找
        return self.hand[max_index]
    
    def play_drawed(self, card: Card) -> bool:
        """ 判斷是否打掉這張抽來的牌 """
        if (not Rule.play_after_draw): return False
        return self.agent.play_drawed(card)
