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
        """ 在所有合法動作中均勻隨機 """
        prob_vector: list[float] = [0.0] * 61
        for i in self.legal_ids(): prob_vector[i] = 1.0
        if (self.draw_allowed()): prob_vector[60] = 1.0
        return tuple(self.softmax(prob_vector))
