from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from torch import Tensor
from core import Card
import weakref

if TYPE_CHECKING: from engine import UnoGame

class Agent(ABC):
    def __init__(
        self, 
        eta: float = 0.85, 
        p_network_path: str|None = None, 
        q_network_path: str|None = None, 
        replay_buffer_path: str|None = None, 
        reservior_memory_path: str|None = None, 
        inference_only: bool = False
    ) -> None:
        """ 設定 eta 以初始化 Agent """
        self.eta: float = eta
        self.q_net_path   : str|None = q_network_path
        self.p_net_path   : str|None = p_network_path
        self.q_memory_path: str|None = replay_buffer_path
        self.p_memory_path: str|None = reservior_memory_path
        self.inference_only: bool = inference_only

    def save(self) -> None: pass
    
    def eval(self) -> None: pass
    
    def train(self) -> None: pass
    
    def push_p(self, state: Tensor, card: Card|None) -> float|None: pass

    def push_q(self, state: Tensor, card: Card|None, reward: float, next_state: Tensor, done: bool) -> float|None: pass
    
    def reload(self, batch_size: int = 64, gamma: float = 0.99, lr: float = 0.001) -> None: pass
    
    def softmax(self, prob_vec: list[float], prob_sum: float|None = None) -> list[float]:
        """ softmax 61 維度的機率向量 """
        # 自己計算總和
        if (prob_sum is None):
            prob_sum = 0
            for prob in prob_vec: prob_sum += prob
        # 總和為零 -> Error
        if (prob_sum <= 0): raise ValueError("prob_sum==0 in Agent::softmax")
        # 真正的 softmax
        for i in range(len(prob_vec)): prob_vec[i] /= prob_sum
        return prob_vec

    def can_play(self, card: Card) -> bool:
        """ 判斷現在能否打這張 """
        in_hand : bool = self.game.curr_player.has(card)
        playable: bool = self.game.deck.is_playable(card)
        return (in_hand and playable)

    def legal_ids(self) -> list[int]:
        """ 高效率取得當前手牌中所有合法出牌的 ID 列表 """
        ids: list[int] = []
        deck = self.game.deck
        seen_color: set[int] = set()
        seen_black: set[int] = set()
        for card in self.game.curr_player.hand:
            if card.is_black():
                val: int = card.value.value
                if (val in seen_black): continue
                # 黑牌對應 4 種顏色變化
                seen_black.add(val)
                if (deck.is_playable(card)):
                    ids.extend(val*4+col for col in range(4))
            else:
                if (card.id in seen_color): continue
                seen_color.add(card.id)
                if deck.is_playable(card): ids.append(card.id)          
        return sorted(ids)

    def get_tensor(self) -> Tensor:
        """ 取得遊戲狀態張量 """
        return self.game.get_tensor()
    
    def set_game(self, game: 'UnoGame') -> None:
        """ get weak reference of game """
        if hasattr(self, 'game'): return
        self.game: 'UnoGame' = weakref.proxy(game)
        self.reload()
        self.eval()
    
    @abstractmethod
    def get_prob_vector(self) -> tuple[float, ...]:
        """ 輸出出各種牌的機率
            回傳一個出牌機率向量
            索引值對應到牌的種類
            最後一個元素代表抽牌
            註: 此函數會自動 softmax
            註: 輸出不為 0 者必定合法
        """
        ...
