from core.card import Card
from agent import Agent

class HumanAgent(Agent):
    """ 人類操控的Agent """
    def gen_prob_vector(self) -> tuple[float, ...]:
        """ TODO: 讓人類輸入要打的牌 """
        text: str = input("請輸入要打的牌(如: R +4): ")
        
