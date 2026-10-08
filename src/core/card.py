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

# 牌張
class Card:
    def __init__(self, color: Color, value: Value) -> None:
        """ 初始化牌張 """
        self.id: int = color.value + value.value*4

    @classmethod
    def from_id(cls, id: int) -> Card:
        """ 使用 id 初始化牌張 """
        color: Color = Color(id %  4)
        value: Value = Value(id // 4)
        return cls(color, value)

    @property
    def value(self) -> Value: return Value(self.id // 4)
    @value.setter
    def value(self, value: Value) -> None: self.id = self.color.value + value.value*4
    
    @property
    def color(self) -> Color: return Color(self.id % 4)
    @color.setter
    def color(self, color: Color) -> None: self.id = color.value + self.value.value*4
    
    def __lt__(self, other: Card) -> bool:
        if (self.is_black()):           return self.id < other.id
        if (self.color != other.color): return self.color.value < other.color.value
        return self.value.value < other.value.value
    
    def __str__(self) -> str: 
        if (self.is_black()): return f"{self.value}"
        return f"{self.color}|{self.value}"
    
    def __repr__(self) -> str: return self.__str__()

    def is_black(self) -> bool:
        """ 判斷是否為黑色牌 """
        return self.value.is_black()

    def is_color(self) -> bool:
        """ 判斷是否為顏色牌 """
        return not self.value.is_black()

    def is_playable_after(self, deck_top: Card, has_effect: bool = False) -> bool:
        """ 判斷這張牌是否可以打 """
        # 連續禁止
        if (deck_top.value == Value.skip and has_effect):
            return (Rule.stack_skip and self.value == Value.skip)
        # +2, +4
        if (deck_top.value == Value.plus2 and has_effect):
            if (Rule.stack_plus2 and self.value == Value.plus2): return True
            if (Rule.stack_all_plus and self.value == Value.plus4): return True
            return False
        if (deck_top.value == Value.plus4 and has_effect):
            if (Rule.stack_plus4 and self.value == Value.plus4): return True
            if (Rule.stack_all_plus and self.value == Value.plus2 and self.color == deck_top.color): return True
            return False
        # 隨時可打
        if (self.is_black()): return True
        # 其他的牌
        return (self.color == deck_top.color or self.value == deck_top.value)
