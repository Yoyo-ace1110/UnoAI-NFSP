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
        self.training: bool = False
        # NFSP: 上一個動作是否由「最佳回應(Q 網路)」模式產生。只有這種動作才該存進 SL 記憶庫。
        # 非 NFSP 的 Agent (Classic / Random / Human) 視為示範者，一律為 True。
        self.last_step_is_br: bool = True

    def save(self) -> None: pass
    
    def eval(self) -> None: self.training = False
    
    def train(self) -> None: self.training = True
    
    def push_p(self, state: Tensor, card: Card|None, mask: Tensor|None = None) -> float|None: pass

    def push_q(
        self, 
        state: Tensor, 
        card: Card|None, 
        reward: float, 
        next_state: Tensor, 
        done: bool, 
        next_mask: Tensor|None = None,
        mask: Tensor|None = None
    ) -> float|None: pass
    
    def reload(self, batch_size: int = 64, gamma: float = 0.99, lr: float = 1e-3) -> None: pass
    
    def softmax(self, prob_vec: list[float], prob_sum: float|None = None) -> list[float]:
        """ 將 61 維度的權重向量正規化成機率 """
        # 自己計算總和
        if (prob_sum is None): prob_sum = sum(prob_vec)
        # 總和為零 -> Error
        if (prob_sum <= 0): raise ValueError("prob_sum==0 in Agent::softmax")
        for i in range(len(prob_vec)): prob_vec[i] /= prob_sum
        return prob_vec

    def legal_ids(self) -> list[int]:
        """ 目前所有合法的牌 id (已快取；只掃手牌，不掃 60 個 id) """
        return self.game.legal_ids()

    def draw_allowed(self) -> bool:
        """ 目前是否可以選擇抽牌 """
        return self.game.draw_allowed()

    def can_play(self, card: Card) -> bool:
        """ 判斷現在能否打這張 """
        in_hand : bool = self.game.curr_player.has(card)
        playable: bool = self.game.deck.is_playable(card)
        return (in_hand and playable)

    def set_game(self, game: 'UnoGame') -> None:
        """ get weak reference of game """
        if hasattr(self, 'game'): return
        self.game: 'UnoGame' = weakref.proxy(game)
        self.reload()
        self.eval()
    
    def get_tensor(self) -> Tensor:
        """ 取得遊戲狀態張量 """
        return self.game.get_tensor()
    
    @abstractmethod
    def get_prob_vector(self) -> tuple[float, ...]:
        """ 輸出出各種牌的機率
            回傳一個出牌機率向量
            索引值對應到牌的種類
            最後一個元素代表抽牌
            註: 輸出不為 0 者必定合法
        """
        ...
