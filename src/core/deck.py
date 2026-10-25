import random
from .card import Card, Value, _COLORS

# 標準 Uno 牌組張數
DECK_SIZE: int = 108

# 牌堆
class Deck:
    def __init__(self, out: bool = True) -> None:
        """ 初始化牌堆 (標準 108 張，每張都是獨立的 Card 物件) """
        self.out: bool = out
        self.cards: list[Card] = []
        self.played: list[Card] = []
        # 加入 1 張 0 (每色一張)
        for color in _COLORS: 
            self.cards.append(Card(color, Value(0)))
        # 加入 1~9 各兩張 (每色兩張)
        for i in range(1, 10):
            for color in _COLORS:
                for _ in range(2): self.cards.append(Card(color, Value(i)))
        # 翻出第一張牌 (只會是數字牌)
        choice: int = random.randrange(len(self.cards))
        self.top: Card = self.cards.pop(choice)
        # 加入功能牌 (turn, skip, +2 每色兩張)
        for i in range(10, 13):
            for color in _COLORS:
                for _ in range(2): self.cards.append(Card(color, Value(i)))
        # 加入黑色牌 (wild, +4 各四張)
        for i in range(13, 15):
            for color in _COLORS:
                self.cards.append(Card(color, Value(i)))
        # 效果處理
        self.has_effect: bool = False
        assert (len(self.cards) + 1 == DECK_SIZE)

    def size(self) -> int:
        """ 牌堆剩餘幾張牌 """
        return len(self.cards)

    def empty(self) -> bool:
        """ 判斷牌堆是否為空 """
        return (self.size() == 0)
    
    def _restore_black(self) -> None:
        """ 黑牌被打出時顏色會被改成玩家選擇的顏色，洗回牌堆前還原成每色各一張 """
        counts: dict[Value, int] = {Value.wild: 0, Value.plus4: 0}
        for card in self.cards:
            if (not card.is_black()): continue
            value: Value = card.value
            card.color = _COLORS[counts[value] % 4]
            counts[value] += 1

    def reset(self) -> None:
        """ 重置牌庫 (呼叫前，所有玩家的手牌應該已經歸還) """
        self.cards.extend(self.played)
        self.cards.append(self.top)
        self.played.clear()
        self._restore_black()
        self.shuffle()
        # 抽出底牌 (數字牌)
        for i in range(len(self.cards)-1, -1, -1):
            if (self.cards[i].is_number()):
                self.top = self.cards.pop(i)
                break
        else:
            self.top = self.cards.pop()
        self.has_effect = False

    def shuffle(self) -> None:
        """ 牌堆洗牌 """
        random.shuffle(self.cards)
    
    def reshuffle(self) -> None:
        if (self.out): print("deck was reshuffled")
        self.cards.extend(self.played)
        self.played.clear()
        self._restore_black()
        self.shuffle()
    
    def set_top(self, card: Card) -> None:
        """ 更新牌頂 """
        self.played.append(self.top)
        self.top = card
    
    def draw_one(self) -> Card|None:
        """ 抽出一張牌 """
        if (self.empty()): self.reshuffle()
        if (self.empty()): return None
        return self.cards.pop()

    def draw(self, n: int) -> list[Card]:
        """ 抽出 n 張牌 """
        result: list[Card] = []
        for _ in range(n):
            drawed: Card|None = self.draw_one()
            if (drawed is not None): result.append(drawed)
        return result

    def is_playable(self, card: Card) -> bool:
        """ 判斷牌能不能打 """
        return card.is_playable_after(self.top, self.has_effect)
