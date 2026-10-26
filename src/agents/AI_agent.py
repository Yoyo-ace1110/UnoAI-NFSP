from models import DQNTrainer, PolicyTrainer, QNetwork, PNetwork, DEVICE
from random import random, choice
from .agent import Agent
from torch import Tensor
from core import Card
import copy
import torch

class AIAgent(Agent):
    def __init__(
        self, 
        eta: float = 0.85, 
        p_network_path: str|None = None, 
        q_network_path: str|None = None, 
        replay_buffer_path: str|None = None, 
        reservior_memory_path: str|None = None, 
        inference_only: bool = False,
        epsilon: float = 0.3,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100000,
        update_every: int = 8,
        warmup: int = 1000,
        cql_alpha: float = 0.2,
        double_dqn: bool = True,
    ) -> None:
        """ AI Agent --version 1
            eta: 使用 Q-Network 的機率, 其餘使用 Policy Network
            inference_only: 只做推論, 不建立 trainer, 不讀記憶庫
            epsilon: 訓練時, Q-Network 的初始隨機探索機率 (ε-greedy)
            epsilon_decay_steps: ε 在 epsilon_decay_steps 內線性衰減
            epsilon_end: ε 經過線性衰減到 epsilon_end
            update_every: 每新增幾筆資料做一次梯度更新
            warmup: 記憶庫至少要有幾筆資料才開始訓練
        """
        super().__init__(
            eta, 
            p_network_path, 
            q_network_path, 
            replay_buffer_path, 
            reservior_memory_path,
            inference_only
        )
        self.epsilon: float = epsilon
        self.epsilon_end: float = epsilon_end
        self.epsilon_decay_steps: int = max(1, epsilon_decay_steps)
        self.br_steps: int = 0      # 已經做過幾次「最佳回應」決策 (用來計算 ε)
        self.update_every: int = update_every
        self.warmup: int = warmup
        self.cql_alpha: float = cql_alpha
        self.double_dqn: bool = double_dqn
        self._q_net: QNetwork|None = None   # 僅 inference_only 使用
        self._p_net: PNetwork|None = None
        # 「選動作」專用的網路 (放 CPU): batch=1 的小網路在 GPU 上反而慢
        self._act_q: QNetwork
        self._act_p: PNetwork

    @property
    def q_net(self) -> QNetwork: 
        return (self._q_net if self.inference_only else self.q_trainer.q_net)  # type: ignore
    
    @property
    def p_net(self) -> PNetwork: 
        return (self._p_net if self.inference_only else self.p_trainer.p_net)  # type: ignore

    @property
    def current_epsilon(self) -> float:
        """ 目前的探索率 (線性衰減) """
        frac: float = min(self.br_steps / self.epsilon_decay_steps, 1.0)
        return self.epsilon + (self.epsilon_end - self.epsilon) * frac

    def eval(self) -> None:
        """ 評估模式 """
        super().eval()
        self.q_net.eval()
        self.p_net.eval()
        
    def train(self) -> None:
        """ 訓練模式 """
        super().train()
        self.q_net.train()
        self.p_net.train()

    def reload(self, batch_size: int = 64, gamma: float = 0.99, lr: float = 1e-3) -> None:
        """ 重新載入模型 """
        state_dim: int = self.game.state_dim
        if (self.q_net_path is None): self.q_net_path = "q_network_v1.bin"
        if (self.p_net_path is None): self.p_net_path = "p_network_v1.bin"
        if (self.q_memory_path is None): self.q_memory_path = "ReplayBuffer_v1.bin"
        if (self.p_memory_path is None): self.p_memory_path = "ReservoirMemory_v1.bin"
        # 評估模式: 只載入網路權重一次
        if (self.inference_only):
            q_net: QNetwork = QNetwork(state_dim, self.q_net_path)
            p_net: PNetwork = PNetwork(state_dim, self.p_net_path)
            if (not (q_net.load() and p_net.load())):
                raise RuntimeError(f"無法載入權重: {self.q_net_path}, {self.p_net_path}")
            self._q_net = q_net.cpu().eval()
            self._p_net = p_net.cpu().eval()
            self._act_q, self._act_p = self._q_net, self._p_net
            return
        self.p_trainer: PolicyTrainer = PolicyTrainer(
            state_dim, 
            lr, 
            batch_size, 
            self.p_net_path, 
            self.p_memory_path, 
            self.update_every, 
            self.warmup
        )
        self.q_trainer: DQNTrainer = DQNTrainer(
            state_dim, 
            lr, 
            gamma, 
            batch_size, 
            self.q_net_path, 
            self.q_memory_path, 
            self.update_every, 
            self.warmup, 
            cql_alpha=self.cql_alpha, double_dqn=self.double_dqn
        )
        self.q_trainer.load()
        self.p_trainer.load()
        if (DEVICE.type == "cpu"): 
            self._act_q = self.q_trainer.q_net
            self._act_p = self.p_trainer.p_net
        else:
            self._act_q = copy.deepcopy(self.q_trainer.q_net).cpu().eval()
            self._act_p = copy.deepcopy(self.p_trainer.p_net).cpu().eval()
    
    def push_p(self, state: Tensor, card: Card|None, mask: Tensor|None = None) -> float|None: 
        loss: float|None = self.p_trainer.push(state, card, mask)
        if ((loss is not None) and (DEVICE.type != "cpu")): 
            self._act_p.load_state_dict(self.p_net.state_dict())
        return loss

    def push_q(
        self, 
        state: Tensor, 
        card: Card|None, 
        reward: float, 
        next_state: Tensor, 
        done: bool, 
        next_mask: Tensor|None = None,
        mask: Tensor|None = None
    ) -> float|None: 
        loss: float|None = self.q_trainer.push(state, card, reward, next_state, done, next_mask, mask)
        if ((loss is not None) and (DEVICE.type != "cpu")): 
            self._act_q.load_state_dict(self.q_net.state_dict())
        return loss

    def save(self) -> None:
        if (self.inference_only): return
        self.q_trainer.save()
        self.p_trainer.save()

    def get_prob_vector(self) -> tuple[float, ...]:
        """ 根據 eta 切換模型並預測 """
        with torch.no_grad():
            if (random() > self.eta): 
                self.last_step_is_br = False
                return self._policy()
            self.last_step_is_br = True
            return self._quality()

    def _quality(self) -> tuple[float, ...]:
        """ predict by Q-Network (訓練時為 epsilon-greedy) """
        vector: list[float] = [0.0] * 61
        if (self.training):
            epsilon: float = self.current_epsilon
            self.br_steps += 1
        else: epsilon = 0.0
        if (random() < epsilon):
            ids: list[int] = list(self.game.legal_ids())
            if (self.game.draw_allowed()): ids.append(60)
            vector[choice(ids)] = 1.0
            # 探索用的隨機動作只給 Q 學習
            self.last_step_is_br = False
            return tuple(vector)
        mask: Tensor = self.game.legal_mask()
        q_values: Tensor = self._act_q(self.get_tensor().unsqueeze(0)).squeeze(0)
        # 非法動作設為 -inf, 只留下 Q 值最大者
        vector[int(q_values.masked_fill(~mask, float("-inf")).argmax())] = 1.0
        return tuple(vector)
    
    def _policy(self) -> tuple[float, ...]:
        """ predict by Policy Network (平均策略) """
        mask: Tensor = self.game.legal_mask()
        logits: Tensor = self._act_p(self.get_tensor().unsqueeze(0)).squeeze(0)
        # 先遮蔽非法動作再 softmax, 輸出已是合法的機率分佈
        probs: list[float] = torch.softmax(logits.masked_fill(~mask, float("-inf")), dim=-1).tolist()  # type: ignore
        return tuple(probs)
