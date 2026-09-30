from __future__ import annotations
from rule import Rule
from enum import Enum

# 顏色
class Color(Enum):
    Red    = 0
    Blue   = 1
    Green  = 2
    Yellow = 3
    
    def __repr__(self) -> str:
        string: tuple[str, str, str, str]
        string = ("R", "B", "G", "Y" )
        return string[self.value]

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

    def is_number(self) -> bool:
        """ 判斷是否為數字 """
        return self.value <= 9

    def is_black(self) -> bool:
        """ 判斷是否為黑色牌 """
        is_wild:  bool = (self.value == 13)
        is_plus4: bool = (self.value == 14)
        return is_wild or is_plus4

    def __repr__(self) -> str:
        string: tuple[str, str, str, str, str]
        if (self.is_number()): return str(self.value)
        string = ("turn", "skip", "wild", "+2", "+4")
        return str(string[self.value-10])

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
    def value(self, value: Value) -> None: self.value = value
    
    @property
    def color(self) -> Color: return Color(self.id % 4)
    @color.setter
    def color(self, color: Color) -> None: self.color = color

    def __repr__(self) -> str: return f"{self.color}|{self.value}"

    def is_black(self) -> bool:
        """ 判斷是否為黑色牌 """
        return self.value.is_black()

    def is_color(self) -> bool:
        """ 判斷是否為顏色牌 """
        return not self.value.is_black()

    def is_playable_after(self, card: Card) -> bool:
        """ 判斷這張牌是否可以打 """
        # 連續禁止
        if (card.value == Value.skip):
            return (Rule.stack_skip and self.value == Value.skip)
        # +2, +4
        if (card.value == Value.plus2):
            if (Rule.stack_plus2 and self.value == Value.plus2): return True
            if (Rule.stack_all_plus and self.value == Value.plus4): return True
            return False
        if (card.value == Value.plus4):
            if (Rule.stack_plus4 and self.value == Value.plus4): return True
            if (Rule.stack_all_plus and self.value == Value.plus2 and self.color == card.color): return True
            return False
        # 迴轉/變色/數字
        return (self.color == card.color or self.value == card.value)
