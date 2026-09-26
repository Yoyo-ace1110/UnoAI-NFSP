from enum import Enum

class Card:
    # 牌面
    class ID(Enum):
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
        wild  = 12 # 變色
        plus2 = 13 # +2
        plus4 = 14 # +4    
    # 顏色
    class Color(Enum):
        Red     = 0
        Blue    = 1
        Green   = 2
        Yellow  = 3

    def __init__(self, id: ID, color: Color) -> None:
        """ 初始化牌張 """
        self.code: int = color.value + id.value*4

    @property
    def id(self) -> ID: return self.ID(self.code // 4)
    @id.setter
    def id(self, x: ID) -> None: self.id = x
    
    @property
    def color(self) -> Color: return self.Color(self.code % 4)
    @color.setter
    def color(self, x: Color) -> None: self.color = x

    def is_black(self) -> bool:
        """ 判斷是否為黑色牌 """
        is_wild:  bool = (self.id == self.ID.wild)
        is_plus4: bool = (self.id == self.ID.plus4)
        return is_wild or is_plus4
