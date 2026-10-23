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
    ) -> None:
        super().__init__(
            eta, 
            p_network_path, 
            q_network_path, 
            replay_buffer_path, 
            reservior_memory_path
        )
    
    def get_prob_vector(self) -> tuple[float, ...]:
        """ 根據經典規則輸出出牌機率向量 """
        prob_vector: list[float] = [0.0] * 61
        player: Player = self.game.curr_player
        # 收集玩家手牌中擁有的顏色（用於變色牌策略）
        hand_colors: set[Color] = {c.color for c in player.hand if c.is_color()}
        # 優先級 1: 數字牌
        tier1: list[int] = []
        for i in range(60):
            card: Card = Card.from_id(i)
            if card.value.is_number() and self.can_play(card):
                tier1.append(i)
        if tier1:
            for i in tier1: prob_vector[i] = 1.0
            return tuple(self.softmax(prob_vector))
        # 優先級 2: 迴轉 (turn) 或 禁止 (skip)
        tier2: list[int] = []
        for i in range(60):
            card: Card = Card.from_id(i)
            if card.value in (Value.turn, Value.skip) and self.can_play(card):
                tier2.append(i)
        if tier2:
            for i in tier2: prob_vector[i] = 1.0
            return tuple(self.softmax(prob_vector))
        # 優先級 3: 變色牌 (wild)，優先選擇自己手上有的顏色
        tier3_preferred: list[int] = []
        tier3_fallback: list[int] = []
        for i in range(60):
            card: Card = Card.from_id(i)
            if card.value == Value.wild and self.can_play(card):
                if card.color in hand_colors:
                    tier3_preferred.append(i)
                else:
                    tier3_fallback.append(i)
        if tier3_preferred:
            for i in tier3_preferred: prob_vector[i] = 1.0
            return tuple(self.softmax(prob_vector))
        if tier3_fallback:
            for i in tier3_fallback: prob_vector[i] = 1.0
            return tuple(self.softmax(prob_vector))
        # 優先級 4a: +2
        tier4a: list[int] = []
        for i in range(60):
            card: Card = Card.from_id(i)
            if card.value == Value.plus2 and self.can_play(card):
                tier4a.append(i)
        if tier4a:
            for i in tier4a: prob_vector[i] = 1.0
            return tuple(self.softmax(prob_vector))
        # 優先級 4b: +4
        tier4b: list[int] = []
        for i in range(60):
            card: Card = Card.from_id(i)
            if card.value == Value.plus4 and self.can_play(card):
                tier4b.append(i)
        if tier4b:
            for i in tier4b: prob_vector[i] = 1.0
            return tuple(self.softmax(prob_vector))
        # 預設: 以上皆無，選擇抽牌 (索引 60)
        prob_vector[60] = 1.0
        return tuple(self.softmax(prob_vector))
