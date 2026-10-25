from core import Card, Value, Color
from .agent import Agent

class HumanAgent(Agent):
    """ 人類操控的Agent """
    def __init__(
        self, 
        eta: float = 0.85, 
        p_network_path: str|None = None, 
        q_network_path: str|None = None, 
        replay_buffer_path: str|None = None, 
        reservior_memory_path: str|None = None, 
        inference_only: bool = False
    ) -> None:
        super().__init__(
            eta, 
            p_network_path, 
            q_network_path, 
            replay_buffer_path, 
            reservior_memory_path,
            inference_only
        )
    
    def get_prob_vector(self) -> tuple[float, ...]:
        """ 讓人類輸入要打的牌 """
        while True:
            text: str = input("請輸入要打的牌 (如: R|+4 or draw): ").strip()
            # 解析使用者輸入
            if (text in ["draw", "Draw", "d", "D"]):
                if (self.draw_allowed()): return tuple(float(i==60) for i in range(61))
                print("\nYou have a playable card, you can not draw")
                continue
            try:
                color_str: str
                value_str: str
                color_str, value_str = (text.split("|"))
                id_: int = Card(Color.from_str(color_str), Value.from_str(value_str)).id
            except ValueError:
                print("\nUnable to parse the card")
                continue
            # 檢查輸入是否合法
            if (not self.can_play(Card.from_id(id_))):
                print("\nThis card is not playable")
                continue
            return tuple(float(i==id_) for i in range(61))
