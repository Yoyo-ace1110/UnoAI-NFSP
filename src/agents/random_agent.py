from random import uniform
from .agent import Agent

class RandomAgent(Agent):
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
        """ 回傳一個隨機的機率向量 """
        legal: list[int] = self.legal_ids()
        prob_vector: list[float] = [0.0]*61
        # 抽牌永遠隨機
        prob_vector[60] = uniform(0, 1)
        prob_sum: float = prob_vector[60]
        # 生成隨機向量
        for i in legal:
            prob_vector[i] = uniform(0, 1)
            prob_sum += prob_vector[i]
        return tuple(self.softmax(prob_vector, prob_sum))
