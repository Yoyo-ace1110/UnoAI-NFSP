from core import Card, Value, Color
from .agent import Agent

class HumanAgent(Agent):
    """ 人類操控的Agent """
    def gen_prob_vector(self) -> tuple[float, ...]:
        """ 讓人類輸入要打的牌 """
        text: str = input("請輸入要打的牌(如: R|+4): ").strip()
        if (text == "draw"): id_: int = 60
        else:
            color_str, value_str = (text.split("|"))
            color: Color = Color.from_str(color_str)
            value: Value = Value.from_str(value_str)
            id_ = Card(color, value).id
        return tuple((1 if i==id_ else 0) for i in range(61))
    
    def play_drawed(self, card: Card) -> bool:
        """ 讓人類判斷是否打掉抽到的牌 """
        while (True):
            # text: str = input("Play this card you just drawed? (y/n): ").strip()
            text: str = input("是否打掉這張剛摸到的牌? (y/n): ").strip()
            if (text == "n" or text == "N"): return False
            if (text == "y" or text == "Y"): return True
            print("Unknown cammand, please try again")

    def decide_color_for_black(self) -> Color:
        """ 讓人類判斷打掉黑色牌後變甚麼顏色 """
        text: str = input("請輸入要變的顏色(R/B/G/Y): ")
        return Color.from_str(text.strip())
