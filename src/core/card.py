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

# 列舉表: 使用 tuple 索引取代 Enum(x) 呼叫 (加速)
COLORS: tuple[Color, ...] = tuple(Color)
VALUES: tuple[Value, ...] = tuple(Value)
TURN:  int = Value.turn .value
SKIP:  int = Value.skip .value
PLUS2: int = Value.plus2.value
WILD:  int = Value.wild .value
PLUS4: int = Value.plus4.value

# 牌張
class Card:
    __slots__ = ("id",)

    def __init__(self, color: Color, value: Value) -> None:
        """ 初始化牌張 id[value(4bits)|color(2bits)] """
        self.id: int = color.value + (value.value << 2)

    @classmethod
    def from_id(cls, id: int) -> Card:
        """ 使用 id 初始化牌張 """
        card: Card = cls.__new__(cls)
        card.id = id
        return card

    @property
    def value(self) -> Value: 
        return VALUES[self.id >> 2]
    @value.setter
    def value(self, value: Value) -> None: 
        # (self.id & 0b11) + (value << 2)
        self.id = (self.id & 3) + (value.value << 2)

    @property
    def color(self) -> Color: 
        return COLORS[self.id & 3]
    @color.setter
    def color(self, color: Color) -> None: 
        # 清掉最低兩個位元(顏色)後，再放入新顏色。注意: << 的優先順序比 + 低，所以不能寫成 a + b << 2
        self.id = (self.id & ~0b11) | color.value

    @property
    def sort_key(self) -> int:
        """ 排序用鍵值 """
        value: int = (self.id >> 2)
        # 黑色牌直接把它設為最大 (加上1000)
        if (value >= WILD): return (1000 + self.id)
        return (self.id & 3)*100 + value

    def __lt__(self, other: Card) -> bool: 
        """ 排序根據 sort_key """
        return self.sort_key < other.sort_key

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
        return (self.id >> 2) >= WILD

    def is_color(self) -> bool:
        """ 判斷是否為顏色牌 """
        return (self.id >> 2) < WILD

    def is_number(self) -> bool:
        """ 判斷是否為數字牌 """
        return (self.id >> 2) <= 9

    def is_playable_after(self, deck_top: Card, has_effect: bool = False) -> bool:
        """ 判斷這張牌是否可以打 """
        value: int = self.id >> 2
        top_value: int = deck_top.id >> 2
        if (has_effect):
            # 連續禁止
            if (top_value == SKIP):
                return (Rule.stack_skip and value == SKIP)
            # +2, +4
            if (top_value == PLUS2):
                if (Rule.stack_plus2 and value == PLUS2): return True
                if (Rule.stack_all_plus and value == PLUS4): return True
                return False
            if (top_value == PLUS4):
                if (Rule.stack_plus4 and value == PLUS4): return True
                if (Rule.stack_all_plus and value == PLUS2 and (self.id & 3) == (deck_top.id & 3)): return True
                return False
        # 隨時可打
        if (value >= WILD): return True
        # 其他的牌
        return ((self.id & 3) == (deck_top.id & 3) or value == top_value)

# 其他函數
def to_vector(cards: list[Card]) -> list[int]:
    """ 將一堆牌轉為向量 """
    vector: list[int] = [0] * 60
    for card in cards: 
        vector[card.id] += 1
    return vector
