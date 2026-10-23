from __future__ import annotations
from typing import TYPE_CHECKING
from .agent import Agent
from core import Card, Value, Color

if TYPE_CHECKING: from engine import Player

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
    
    def get_prob_vector(self) -> tuple[float, ...]:
        """ 根據經典規則輸出出牌機率向量 """
        prob_vector: list[float] = [0.0] * 61
        legal: list[int] = self.legal_ids()
        if (not legal):
            # 無牌可出，選擇抽牌
            prob_vector[60] = 1.0
            return tuple(self.softmax(prob_vector))
        player: Player = self.game.curr_player
        hand_colors: set[Color] = {card.color for card in player.hand if card.is_color()}
        # 優先級 1: 數字牌
        tier1 = [i for i in legal if ((i < 60) and Card.from_id(i).is_number())]
        if tier1:
            for i in tier1: prob_vector[i] = 1.0
            return tuple(self.softmax(prob_vector))
        # 優先級 2: 迴轉 (turn) 或 禁止 (skip)
        tier2 = [i for i in legal if ((i < 60) and Card.from_id(i).value in (Value.turn, Value.skip))]
        if tier2:
            for i in tier2: prob_vector[i] = 1.0
            return tuple(self.softmax(prob_vector))
        # 優先級 3: 變色牌 (wild)
        tier3_fallback : list[int] = []
        tier3_preferred: list[int] = []
        for i in legal:
            if (i >= 60): continue
            card: Card = Card.from_id(i)
            if (card != Value.wild): continue
            if (card.color in hand_colors):
                tier3_preferred.append(i)
            tier3_fallback.append(i)
        if tier3_preferred:
            for i in tier3_preferred: prob_vector[i] = 1.0
            return tuple(self.softmax(prob_vector))
        if tier3_fallback:
            for i in tier3_fallback: prob_vector[i] = 1.0
            return tuple(self.softmax(prob_vector))
        # 優先級 4a: +2
        tier4a = [i for i in legal if ((i < 60) and (Card.from_id(i).value == Value.plus2))]
        if tier4a:
            for i in tier4a: prob_vector[i] = 1.0
            return tuple(self.softmax(prob_vector))
        # 優先級 4b: +4
        tier4b = [i for i in legal if ((i < 60) and (Card.from_id(i).value == Value.plus4))]
        if tier4b:
            for i in tier4b: prob_vector[i] = 1.0
            return tuple(self.softmax(prob_vector))
        # this should not happen
        prob_vector[legal[0]] = 1.0
        return tuple(self.softmax(prob_vector))
