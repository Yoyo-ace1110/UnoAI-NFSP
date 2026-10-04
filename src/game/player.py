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

    def remove(self, card: Card) -> None:
        """ 移除這張手牌 """
        self.hand.remove(card)

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

    def softmax_playables(self, deck_top: Card, prob_vec: tuple[float, ...], has_effect: bool = False) -> list[float]:
        """ 輸出手上各牌出牌的機率
            回傳一個出牌機率向量
            索引值對應到手牌索引
            最後一個元素代表抽牌 
        """
        playable_probs: list[float] = []
        color_count: int = self.color_count()
        # 其他牌機率
        for card in self.hand:
            # 檢查是否合法
            is_valid: bool = card.is_playable_after(deck_top, has_effect)
            if (not Rule.black_finisher and self.uno() and card.is_black()): is_valid = False
            if (Rule.color_retention and card.is_color() and color_count == 1): is_valid = False
            # 合法才用 Agent 給的機率, 不合法則填入 0
            if not is_valid:            playable_probs.append(0.0)
            elif (not card.is_black()): playable_probs.append(prob_vec[card.id])
            # 黑牌可以代表該功能的任意一種顏色
            else:
                base_id: int = card.value.value*4
                black_prob: float = max(prob_vec[base_id+i] for i in range(4))
                playable_probs.append(black_prob)
        # 抽牌機率
        playable_probs.append(prob_vec[-1])
        return playable_probs

    def genmove(self, deck_top: Card, has_effect: bool = False) -> Card|None:
        """ 出牌, None代表抽牌 """
        # 取得 Agent 輸出的 61 維決策機率向量
        agent_prob_vec: tuple[float, ...] = self.agent.gen_prob_vector()
        # 傳入 agent_prob_vec 計算合法手牌機率 (長度為 hand_size + 1)
        playable_probs: list[float] = self.softmax_playables(deck_top, agent_prob_vec, has_effect)
        max_index: int = max(range(len(playable_probs)), key=playable_probs.__getitem__)
        # 最後一個元素代表抽牌
        if (max_index == self.hand_size()): return None
        # 其他則去手牌上找
        chosen_card: Card = self.hand[max_index]
        # 遇到黑牌要修正顏色為玩家在 agent_prob_vec 中指定的顏色
        if chosen_card.is_black():
            base_id: int = chosen_card.value.value*4
            chosen_color: int = max(range(4), key=lambda c: agent_prob_vec[base_id+c])
            chosen_card.id = chosen_color + chosen_card.value.value * 4
        return chosen_card

    def play_drawed(self, card: Card) -> bool:
        """ 判斷是否打掉這張抽來的牌 """
        if (not Rule.play_after_draw): return False
        return self.agent.play_drawed(card)
