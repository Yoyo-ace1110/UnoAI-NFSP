from __future__ import annotations
from .agent import Agent
from core import Value

_TURN, _SKIP  = Value.turn.value, Value.skip.value
_PLUS2, _WILD, _PLUS4 = Value.plus2.value, Value.wild.value, Value.plus4.value

class ClassicAgent(Agent):
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

    @staticmethod
    def _pick(ids: list[int]) -> tuple[float, ...]:
        """ 在 ids 中均勻選擇 """
        prob_vector: list[float] = [0.0] * 61
        for i in ids: prob_vector[i] = 1.0 / len(ids)
        return tuple(prob_vector)
    
    def get_prob_vector(self) -> tuple[float, ...]:
        """ 根據經典規則輸出出牌機率向量 """
        ids: list[int] = self.legal_ids()
        # 沒有合法牌 -> 抽牌 (索引 60)
        if (not ids): return self._pick([60])
        # 優先級 1: 數字牌
        tier: list[int] = [i for i in ids if (i >> 2) <= 9]
        if (tier): return self._pick(tier)
        # 優先級 2: 迴轉 (turn) 或 禁止 (skip)
        tier = [i for i in ids if (i >> 2) in (_TURN, _SKIP)]
        if (tier): return self._pick(tier)
        # 優先級 3: 變色牌 (wild), 優先選擇自己手上有的顏色
        tier = [i for i in ids if (i >> 2) == _WILD]
        if (tier):
            hand_colors: set[int] = {c.id & 3 for c in self.game.curr_player.hand if c.is_color()}
            return self._pick([i for i in tier if (i & 3) in hand_colors] or tier)
        # 優先級 4a: +2
        tier = [i for i in ids if (i >> 2) == _PLUS2]
        if (tier): return self._pick(tier)
        # 優先級 4b: +4
        return self._pick([i for i in ids if (i >> 2) == _PLUS4] or ids)
