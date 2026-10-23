from random import uniform
from .agent import Agent
from core import Card

class RandomAgent(Agent):
    def __init__(
        self, 
        eta: float = 0.85, 
        p_network_path: str|None = None, 
        q_network_path: str|None = None, 
        replay_buffer_path: str|None = None, 
        reservior_memory_path: str|None = None, 
    ) -> None:
        super().__init__(
            eta, 
            p_network_path, 
            q_network_path, 
            replay_buffer_path, 
            reservior_memory_path
        )
    
    def get_prob_vector(self) -> tuple[float, ...]:
        """ 回傳一個隨機的機率向量 """
        prob_vector: list[float] = []
        prob_sum: float = 0.0
        # 生成隨機向量
        for i in range(61):
            this_prob: float = 0.0
            # 合法動作才隨機取 0-1 之間的機率
            if (i == 60 or self.can_play(Card.from_id(i))):
                this_prob = uniform(0, 1)
            prob_vector.append(this_prob)
            prob_sum += this_prob
        return tuple(self.softmax(prob_vector, prob_sum))
