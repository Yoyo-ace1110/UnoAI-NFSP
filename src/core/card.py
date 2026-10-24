from __future__ import annotations
from .rule import Rule
from enum import Enum

# 顏色
class Color(Enum):
    Red    = 0
    Blue   = 1
    Green  = 2
    Yellow = 3
    
    @classmethod
    def from_str(cls, text: str) -> Color:
        """ 由使用者輸入建構 Color """
        red_str: tuple[str, ...] = ("R", "r", "Red", "red")
        blue_str: tuple[str, ...] = ("B", "b", "Blue", "blue")
        green_str: tuple[str, ...] = ("G", "g", "Green", "green")
        yellow_str: tuple[str, ...] = ("Y", "y", "Yellow", "yellow")
        if (text in red_str): return cls.Red
        if (text in blue_str): return cls.Blue
        if (text in green_str): return cls.Green
        if (text in yellow_str): return cls.Yellow
        raise ValueError("Unable to parse the Color")

    def __str__(self) -> str:
        string: tuple[str, ...] = ("R", "B", "G", "Y" )
        return string[self.value]

    def __repr__(self) -> str: return self.__str__()
    
# 牌面
class Value(Enum):
    # 數字牌
    num0 = 0 
    num1 = 1
    num2 = 2
    num3 = 3
    num4 = 4
    num5 = 5
    num6 = 6
    num7 = 7
    num8 = 8
    num9 = 9
    # 功能牌
    turn  = 10 # 迴轉
    skip  = 11 # 禁止
    plus2 = 12 # +2
    wild  = 13 # 變色
    plus4 = 14 # +4   

    @classmethod
    def from_str(cls, text: str) -> Value:
        """ 由使用者輸入建構 Value """
        string: tuple[str, ...] = ("turn", "skip", "+2", "wild", "+4")
        if (text.isdecimal()): return cls(int(text))
        return cls(string.index(text)+10)

    def is_number(self) -> bool:
        """ 判斷是否為數字 """
        return self.value <= 9

    def is_black(self) -> bool:
        """ 判斷是否為黑色牌 """
        is_wild:  bool = (self.value == 13)
        is_plus4: bool = (self.value == 14)
        return is_wild or is_plus4

    def __str__(self) -> str:
        string: tuple[str, ...] = ("turn", "skip", "+2", "wild", "+4")
        if (self.is_number()): return str(self.value)
        return str(string[self.value-10])

    def __repr__(self) -> str: return self.__str__()

# 列舉表：用 tuple 索引取代 Enum(x) 呼叫（Enum 建構在 Python 裡很慢，是原本的主要熱點）
_COLORS: tuple[Color, ...] = tuple(Color)
_VALUES: tuple[Value, ...] = tuple(Value)
_TURN:  int = Value.turn.value
_SKIP:  int = Value.skip.value
_PLUS2: int = Value.plus2.value
_WILD:  int = Value.wild.value
_PLUS4: int = Value.plus4.value

# 牌張
class Card:
    __slots__ = ("id",)

    def __init__(self, color: Color, value: Value) -> None:
        """ 初始化牌張 """
        self.id: int = color.value + value.value*4

    @classmethod
    def from_id(cls, id: int) -> Card:
        """ 使用 id 初始化牌張 (不經過 Enum，速度快) """
        card: Card = cls.__new__(cls)
        card.id = id
        return card

    @property
    def value(self) -> Value: return _VALUES[self.id >> 2]
    @value.setter
    def value(self, value: Value) -> None: self.id = (self.id & 3) + value.value*4

    @property
    def color(self) -> Color: return _COLORS[self.id & 3]
    @color.setter
    def color(self, color: Color) -> None: self.id = color.value + (self.id >> 2)*4

    @property
    def sort_key(self) -> int:
        """ 排序用鍵值: 顏色牌依 (顏色, 牌面)，黑牌一律排在最後 (Player.is_all_black 依賴此性質) """
        value: int = self.id >> 2
        if (value >= _WILD): return 1000 + self.id
        return (self.id & 3)*100 + value

    def __lt__(self, other: Card) -> bool: return self.sort_key < other.sort_key

    def __eq__(self, other: object) -> bool:
        """ 透過卡片 id 判斷兩張牌是否相等 """
        if not isinstance(other, Card): return NotImplemented
        return (self.id == other.id)

    def __hash__(self) -> int: return hash(self.id)

    def __str__(self) -> str:
        if (self.is_black()): return f"{self.value} "
        return f"{self.color}|{self.value} "

    def __repr__(self) -> str: return self.__str__()

    def cout(self, show_color_for_black: bool = True, end: str = "\n") -> None:
        """ 
        輸出牌張 color_for_black 帶表黑色牌是否要顯示顏色
        比如說打 只是輸出手牌  color_for_black is False
        比如說打 +4 後變成黃色 color_for_black is True
        此輸出方式包含換行!!!
        """
        if (self.is_black() and not show_color_for_black):
            return print(f"{self.value}", end=end)
        print(f"{self.color}|{self.value}", end=end)

    def is_black(self) -> bool:
        """ 判斷是否為黑色牌 """
        return (self.id >> 2) >= _WILD

    def is_color(self) -> bool:
        """ 判斷是否為顏色牌 """
        return (self.id >> 2) < _WILD

    def is_number(self) -> bool:
        """ 判斷是否為數字牌 """
        return (self.id >> 2) <= 9

    def is_playable_after(self, deck_top: Card, has_effect: bool = False) -> bool:
        """ 判斷這張牌是否可以打 """
        value: int = self.id >> 2
        top_value: int = deck_top.id >> 2
        if (has_effect):
            # 連續禁止
            if (top_value == _SKIP):
                return (Rule.stack_skip and value == _SKIP)
            # +2, +4
            if (top_value == _PLUS2):
                if (Rule.stack_plus2 and value == _PLUS2): return True
                if (Rule.stack_all_plus and value == _PLUS4): return True
                return False
            if (top_value == _PLUS4):
                if (Rule.stack_plus4 and value == _PLUS4): return True
                if (Rule.stack_all_plus and value == _PLUS2 and (self.id & 3) == (deck_top.id & 3)): return True
                return False
        # 隨時可打
        if (value >= _WILD): return True
        # 其他的牌
        return ((self.id & 3) == (deck_top.id & 3) or value == top_value)

# 其他函數
def to_vector(cards: list[Card]) -> list[int]:
    """ 將一堆牌轉為向量 """
    vector: list[int] = [0] * 60
    for card in cards: 
        vector[card.id] += 1
    return vector
