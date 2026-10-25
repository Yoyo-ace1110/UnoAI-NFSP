from core import Rule, Card, Value, Deck, DECK_SIZE
from models import DQNTrainer, PolicyTrainer, print_device_info
from agents import Agent, AIAgent, RandomAgent, HumanAgent
from .player import Player
from torch import Tensor
from dataclasses import dataclass
import random
import torch
import time
import os

class ModelInfo:
    def __init__(
        self, 
        name: str, 
        size: int, 
        p_net: str, 
        q_net: str, 
        p_memory: str, 
        q_memory: str, 
        matches: int = 0, 
        elo: float = 0
    ) -> None:
        self.name: str = name
        self.p_net: str = p_net
        self.q_net: str = q_net
        self.p_memory: str = p_memory
        self.q_memory: str = q_memory
        self.size: int = size
        self.elo: float = elo
        self.matches: int = matches

    def __getitem__(self, key: str): return getattr(self, key)

    def __setitem__(self, key: str, value: str|int|float) -> None: setattr(self, key, value)

class ELO_table:
    def __init__(self, table_path: str = "./ELO_table.txt") -> None: 
        """ 讀取或初始化 ELO 積分表
        結構: dict[name, ModelInfo]
        """
        self.path: str = table_path
        self.table: dict[str, ModelInfo] = {}
        self.load()

    def load(self) -> None:
        """ 嘗試載入積分表 """
        self.table.clear()
        if (not os.path.exists(self.path)): return
        with open(self.path, "r", encoding="utf-8") as file:
            lines: list[str] = file.readlines()
            for line in lines:
                if ("---" in line):        continue
                if ("|" not in line):      continue
                if ("Model Name" in line): continue
                parts = [part.strip() for part in line.split("|")]
                # 欄位不足則略過
                if (len(parts) < 8): continue
                name: str = parts[0]
                p_net: str = parts[1]
                q_net: str = parts[2]
                p_memory: str = parts[3]
                q_memory: str = parts[4]
                size: int = int(parts[5])
                elo: float = float(parts[6])
                matches: int = int(parts[7])
                self.table[name] = ModelInfo(
                    name=name,
                    p_net=p_net,
                    q_net=q_net,
                    p_memory=p_memory,
                    q_memory=q_memory,
                    size=size,
                    elo=elo,
                    matches=matches
                )

    def save(self) -> None:
        """ 儲存 ELO 積分表 """
        with open(self.path, "w", encoding="utf-8") as file:
            file.write("=== UnoAI ELO 排行 ===\n")
            file.write(f"{'Model Name'  :<30}|")
            file.write(f"{'p_net'       :<30}|")
            file.write(f"{'q_net'       :<30}|")
            file.write(f"{'p_memory'    :<30}|")
            file.write(f"{'q_memory'    :<30}|")
            file.write(f"{'Size'        :<10}|")
            file.write(f"{'ELO Rating'  :<10}|")
            file.write(f"{'Matches'     :<8}\n")
            file.write("-" * 195 + "\n")
            for item in self.sorted():
                file.write(f"{str(item.name):<30}|")
                file.write(f"{str(item.p_net):<30}|")
                file.write(f"{str(item.q_net):<30}|")
                file.write(f"{str(item.p_memory):<30}|")
                file.write(f"{str(item.q_memory):<30}|")
                file.write(f"{int(item.size):<10}|")
                file.write(f"{float(item.elo):<10.2f}|")
                file.write(f"{int(item.matches):<8}\n")
        print(f"ELO 積分表已儲存至 {self.path}")

    def sorted(self) -> list[ModelInfo]:
        """ 排序 ELO 積分表 """
        def get_elo(item: ModelInfo) -> float: 
            return float(item.elo)
        return sorted(self.table.values(), key=get_elo, reverse=True)

    def __getitem__(self, key: str) -> ModelInfo: return self.table[key]

    def __setitem__(self, key: str, value: ModelInfo) -> None: self.table[key] = value

    def __contains__(self, key: str) -> bool: return key in self.table

# 獎懲設定
WIN_REWARD:   float = +1.0   # 贏牌
LOSE_REWARD:  float = -1.0   # 輸牌
PLAY_REWARD:  float = +0.025 # 打出手牌
DRAW_PENALTY: float = -0.025 # 主動抽牌

# PLAY_REWARD:  float = +0.01  # 打出手牌
# DRAW_PENALTY: float = -0.01  # 主動抽牌

# 狀態向量: [牌堆張數(1)|廢牌(60)|手牌(60)|牌頂顏色(4)|牌頂牌面(15)|牌頂效果(1)|累加張數(1)|方向(1)|各家張數(玩家數量)]
STATE_BASE: int = 1 + 60 + 60 + 4 + 15 + 1 + 1 + 1
HAND_NORM:  float = 20.0

@dataclass(slots=True)
class Step:
    """ 一個決策點 """
    player: int         # 玩家編號
    agent:  Agent       # 決策的 Agent
    state:  Tensor      # 決策前的狀態 (該玩家視角)
    mask:   Tensor      # 決策前的合法動作遮罩 (61)
    card:   Card|None   # 動作 (None 代表抽牌)
    reward: float       # 即時獎懲 (不含終局 ±1)
    is_br:  bool        # 此動作是否由最佳回應(Q)模式產生

class UnoGame:
    def __init__(self, agents: list[Agent]) -> None:
        """ 初始化整個遊戲 """
        if (not Rule.is_valid()):
            raise ValueError("Invalid Rule") 
        if (len(agents)*Rule.hand_size > DECK_SIZE - 1):
            raise ValueError("Too much player or initial hand size is too big")
        self.turn : int = 0
        self.direc: int = 1
        self.plus_sum: int = 0
        self.deck: Deck = Deck()
        self.elo_table: ELO_table
        self.agents: list[Agent] = agents
        self.players: list[Player] = []
        # 快取 (每次遊戲狀態改變時 invalidate)
        self._state: Tensor|None = None
        self._legal_ids: list[int]|None = None
        self._legal_mask: Tensor|None = None
        # 建立玩家並發牌
        self._setup_players(agents)

    def _setup_players(self, agents: list[Agent]) -> None:
        """ 以新的 agents 重建玩家並發牌。舊玩家的手牌會先歸還牌堆，避免牌越用越少 """
        for player in self.players: player.reset(self.deck)
        self.agents = agents
        self.players = []
        self.turn  = 0
        self.direc = 1
        self.plus_sum = 0
        self.deck.reset()
        for i, agent in enumerate(agents):
            player: Player = Player(agent)
            self.players.append(player)
            player.deal_hand(self.deck.draw(Rule.hand_size))
            player.set_index(i)
        self.invalidate()
        # 事後設定以避免輸入層的維度不同
        for agent in agents: agent.set_game(self)

    @property
    def curr_player(self) -> Player: return self.players[self.turn]

    @property
    def curr_agent(self) -> Agent: return self.curr_player.agent

    @property
    def players_size(self) -> int: return len(self.players)

    @property
    def state_dim(self) -> int: return STATE_BASE + self.players_size

    def invalidate(self) -> None:
        """ 遊戲狀態改變後，清除快取 """
        self._state = None
        self._legal_ids = None
        self._legal_mask = None

    def turn_next(self) -> None:
        """ 輪到下一個人 """
        self.invalidate()
        if (self.direc == 0): 
            # 兩人玩時打迴轉
            self.direc = 1
            return
        # python: x % k = x - k * [x/k]
        self.turn = (self.turn + self.direc) % self.players_size

    def reverse(self) -> None:
        """ 迴轉 """
        if (self.players_size == 2): 
            # 兩人玩時 迴轉==禁止
            self.direc = 0
        else: 
            self.direc *= -1

    def log(self) -> None:
        """ 輸出遊戲狀態 """
        print(f"\ntop card:", end=" ")
        self.deck.top.cout(True)
        for i in range(self.players_size):
            player: Player = self.players[i]
            # 人類玩家輸出手牌, 其餘輸出張數
            if (isinstance(player.agent, HumanAgent)): player.show_hand()
            else: print(f"Player{i}: {self.players[i].hand_size()} cards")

    def reset(self) -> None:
        """ 重置遊戲狀態 """
        self.turn  = 0
        self.direc = 1
        self.plus_sum = 0
        # 讓玩家歸還牌
        for player in self.players: 
            player.reset(self.deck)
        self.deck.reset()
        # 重新發牌給玩家
        for player in self.players: 
            player.deal_hand(self.deck.draw(Rule.hand_size))
        self.invalidate()

    def game_over(self, winner_index: int) -> None:
        """ 遊戲結束 """
        print(f"Game Over! The Winner is Player{winner_index}")

    def get_tensor(self) -> Tensor:
        """ 根據目前玩家將遊戲轉為向量 (快取：狀態沒變時直接回傳同一個張量，請勿原地修改) """
        if (self._state is not None): return self._state
        n: int = self.players_size
        array: list[float] = [0.0] * (STATE_BASE + n)
        array[0] = self.deck.size() / DECK_SIZE
        for card in self.deck.played: array[1 + card.id] += 1.0
        for card in self.curr_player.hand: array[61 + card.id] += 1.0
        # 牌頂 (含黑牌被指定的顏色)、效果、累加張數、方向 —— 這是判斷「哪些牌能出」最關鍵的資訊
        top: Card = self.deck.top
        array[121 + (top.id & 3)] = 1.0
        array[125 + (top.id >> 2)] = 1.0
        array[140] = 1.0 if (self.deck.has_effect) else 0.0
        array[141] = min(self.plus_sum / 8.0, 4.0)
        array[142] = 1.0 if (self.direc >= 0) else 0.0
        # 各家張數：由自己開始依出牌順序排列 (視角旋轉)，同一個網路坐哪個位置都適用
        for k in range(n): 
            array[STATE_BASE + k] = self.players[(self.turn + k) % n].hand_size() / HAND_NORM
        self._state = torch.tensor(array, dtype=torch.float32)
        return self._state

    def legal_ids(self) -> list[int]:
        """ 目前玩家所有可以打的牌 id (只掃手牌並去重，黑牌展開成 4 種顏色) """
        if (self._legal_ids is not None): return self._legal_ids
        deck: Deck = self.deck
        ids: list[int] = []
        seen_color: set[int] = set()
        seen_black: set[int] = set()
        for card in self.curr_player.hand:
            if (card.is_black()):
                value: int = card.id >> 2
                if (value in seen_black): continue
                seen_black.add(value)
                if (deck.is_playable(card)): ids.extend(value*4 + color for color in range(4))
            else:
                if (card.id in seen_color): continue
                seen_color.add(card.id)
                if (deck.is_playable(card)): ids.append(card.id)
        ids.sort()
        self._legal_ids = ids
        return ids

    def draw_allowed(self) -> bool:
        """ 目前玩家可否選擇抽牌: 沒牌可出、被加牌/禁止時必須能抽；否則依 Rule.allow_voluntary_draw """
        return (Rule.allow_voluntary_draw or self.deck.has_effect or not self.legal_ids())

    def legal_mask(self) -> Tensor:
        """ 61 維布林遮罩 (快取)，True 代表合法動作 """
        if (self._legal_mask is not None): return self._legal_mask
        mask: Tensor = torch.zeros(61, dtype=torch.bool)
        ids: list[int] = self.legal_ids()
        if (ids): mask[ids] = True
        mask[60] = self.draw_allowed()
        self._legal_mask = mask
        return mask

    def decide_to_draw(self, out: bool = True) -> None:
        """ 玩家決定抽牌 """
        self.invalidate()
        top_is_skip : bool = (self.deck.top.value == Value.skip  and self.deck.has_effect)
        top_is_plus2: bool = (self.deck.top.value == Value.plus2 and self.deck.has_effect)
        top_is_plus4: bool = (self.deck.top.value == Value.plus4 and self.deck.has_effect)
        # 被禁止不用抽
        if (top_is_skip): 
            if (out): print(f"Player{self.turn} was skipped")
            self.deck.has_effect = False
            return
        # 被加牌
        if (top_is_plus2 or top_is_plus4): 
            if (out): print(f"Player{self.turn} drawed {self.plus_sum} cards")
            self.curr_player.draw(self.deck, self.plus_sum)
            self.deck.has_effect = False # 加完之後讓牌頂失效
            self.plus_sum = 0 # 重置累加指標
            return
        # 沒牌抽
        drawed: Card|None = self.deck.draw_one()
        if (drawed is None): 
            if (out): print(f"deck is empty, Player{self.turn} drawed nothing")
            return
        # 抽一張
        self.curr_player.receive(drawed)
        if (isinstance(self.curr_player.agent, HumanAgent)):
            if (out): print(f"Player{self.turn} drawed")
            if (out): drawed.cout(False)
            return
        if (out): print(f"Player{self.turn} drawed 1 card")

    def play_as_normal(self, played: Card, out: bool = True) -> None:
        """ 正常出牌 """
        self.invalidate()
        self.deck.set_top(played)
        self.curr_player.remove(played)
        if (out): print(f"Player{self.turn} played", end=" ")
        if (out): played.cout(show_color_for_black=True)
        # 迴轉機制
        if (played.value == Value.turn):
            self.reverse()
        # 禁止機制
        elif (played.value == Value.skip):
            self.deck.has_effect = True
        # 連加機制
        elif (played.value == Value.plus2):
            self.deck.has_effect = True
            self.plus_sum += 2
        elif (played.value == Value.plus4):
            self.deck.has_effect = True
            self.plus_sum += 4
        # Uno 機制
        if (self.curr_player.uno()): 
            if (out): print(f"Player{self.turn} Uno!")

    def _unique_agents(self) -> list[Agent]:
        """ 去除重複的 Agent 實例 (selfplay 兩個座位共用同一個 Agent) """
        return list({id(agent): agent for agent in self.agents}.values())

    def _play_episode(self, max_steps: int) -> tuple[list[Step], int]:
        """ 玩一整局，回傳 (所有決策點, 贏家編號), 截斷贏家則為-1 """
        self.reset()
        steps: list[Step] = []
        while (len(steps) < max_steps):
            p_index: int = self.turn
            agent: Agent = self.curr_agent
            # 這兩個值會被快取，Agent 選動作時直接重複使用，不會重算
            state: Tensor = self.get_tensor()
            mask: Tensor = self.legal_mask()
            forced: bool = (self.deck.has_effect or not self.legal_ids())
            played_card: Card|None = self.curr_player.genmove()
            if (played_card is None):
                if (not self.draw_allowed()): raise ValueError("Bot drew while it has a playable card")
                reward: float = (0.0 if forced else DRAW_PENALTY)
                self.decide_to_draw(out=False)
            elif (not (self.deck.is_playable(played_card) and self.curr_player.has(played_card))):
                raise ValueError("Bot played the Card that is not playable")
            else:
                reward = PLAY_REWARD
                self.play_as_normal(played_card, out=False)
            steps.append(Step(p_index, agent, state, mask, played_card, reward, agent.last_step_is_br))
            if (self.curr_player.is_winner()): return steps, p_index
            self.turn_next()
        return (steps, -1)

    @staticmethod
    def _build_transitions(
        steps: list[Step], 
        winner: int, 
        only_player: int|None = None
    ) -> list[tuple[Step, tuple[float, Tensor, Tensor, bool]|None]]:
        """ 將軌跡轉成 (Step, (reward, next_state, next_mask, done)) 
        - next_state / next_mask 是「同一位玩家下一次要做決策時」的狀態，這樣才符合 Bellman 方程式
        - 終局 ±1 只加在每位玩家的最後一步，done=True；不再把終局獎勵灑在每一步 (折扣交給 gamma)
        - 被截斷的局，每位玩家最後一步沒有 next_state，Q 不使用 (None)，但 SL 仍可使用
        """
        by_player: dict[int, list[Step]] = {}
        for step in steps: by_player.setdefault(step.player, []).append(step)
        ones: Tensor = torch.ones(61, dtype=torch.bool)
        result: list[tuple[Step, tuple[float, Tensor, Tensor, bool]|None]] = []
        for p_idx, player_steps in by_player.items():
            if (only_player is not None and p_idx != only_player): continue
            last: int = len(player_steps) - 1
            for k, step in enumerate(player_steps):
                if (k < last):
                    nxt: Step = player_steps[k+1]
                    result.append((step, (step.reward, nxt.state, nxt.mask, False)))
                elif (winner != -1):
                    final: float = (WIN_REWARD if p_idx == winner else LOSE_REWARD)
                    result.append((step, (step.reward + final, step.state, ones, True)))
                else:
                    result.append((step, None))
        return result

    def selfplay(self, num_episodes: int, max_steps: int = 500, log_every: int = 16, save_every: int = 0) -> None:
        """ 模型自我對戰訓練 (NFSP)
        max_steps: 單局步數上限，超過視為截斷
        log_every: 每幾局輸出一次統計
        save_every: 每幾局存檔一次 (0 代表不中途存檔)
        """
        self.deck.out = False   # 避免不必要的輸出
        print_device_info()     # 輸出一次使用裝置
        agents: list[Agent] = self._unique_agents()
        for agent in agents: agent.train()
        print(f"開始進行 {num_episodes} 局自我對戰")
        total_steps = draw_steps = truncated = seat0_wins = 0
        q_losses: list[float] = []
        p_losses: list[float] = []
        start: float = time.time()
        for episode in range(1, num_episodes+1):
            steps, winner = self._play_episode(max_steps)
            total_steps += len(steps)
            draw_steps  += sum(1 for step in steps if step.card is None)
            truncated   += (winner == -1)
            seat0_wins  += (winner == 0)
            # 存入雙記憶庫(Dual Memory Mechanism)並更新雙網路
            for step, transition in self._build_transitions(steps, winner):
                if (transition is not None):
                    reward, next_state, next_mask, done = transition
                    q_loss: float|None = step.agent.push_q(step.state, step.card, reward, next_state, done, next_mask)
                    if (q_loss is not None): q_losses.append(q_loss)
                # NFSP: 只有最佳回應模式的動作才存入 SL 記憶庫
                if (step.is_br):
                    p_loss: float|None = step.agent.push_p(step.state, step.card, step.mask)
                    if (p_loss is not None): p_losses.append(p_loss)
            if (save_every > 0 and episode % save_every == 0):
                for agent in agents: agent.save()
            # 每 n 局輸出進度提示
            if (episode % log_every == 0 or episode == num_episodes): 
                elapsed: float = time.time() - start
                n: int = (episode - 1) % log_every + 1
                print(
                    f"已完成第 {episode} 局 | {total_steps/n:.1f} 步/局 | 抽牌率 {draw_steps/max(total_steps, 1):.1%} | "
                    f"截斷 {truncated} | 0號位勝率 {seat0_wins/n:.0%} | "
                    f"Q-loss {sum(q_losses)/max(len(q_losses), 1):.4f} | P-loss {sum(p_losses)/max(len(p_losses), 1):.4f} | "
                    f"{elapsed/episode:.2f} 秒/局"
                )
                total_steps = draw_steps = truncated = seat0_wins = 0
                q_losses.clear(); p_losses.clear()

    def crossplay(
        self, 
        num_episodes: int, 
        q_trainer: DQNTrainer, 
        p_trainer: PolicyTrainer, 
        target_player_idx: int|None = 0, 
        max_steps: int = 500, 
        log_every: int = 64
    ) -> None:
        """ 模型交叉對戰訓練 (向 self.agents 中的示範者學習：只收集 target_player_idx 的經驗，None 代表所有座位都是示範者) """
        self.deck.out = False   # 避免不必要的輸出
        print_device_info()     # 輸出一次使用裝置
        for agent in self._unique_agents(): agent.eval()
        print(f"開始進行 {num_episodes} 局交叉對戰")
        total_steps = truncated = target_wins = 0
        q_losses: list[float] = []
        p_losses: list[float] = []
        start: float = time.time()
        for episode in range(1, num_episodes+1):
            steps, winner = self._play_episode(max_steps)
            total_steps += len(steps)
            truncated   += (winner == -1)
            target_wins += (winner == target_player_idx or (target_player_idx is None and winner == 0))
            for step, transition in self._build_transitions(steps, winner, only_player=target_player_idx):
                if (transition is not None):
                    reward, next_state, next_mask, done = transition
                    q_loss: float|None = q_trainer.push(step.state, step.card, reward, next_state, done, next_mask)
                    if (q_loss is not None): q_losses.append(q_loss)
                p_loss: float|None = p_trainer.push(step.state, step.card, step.mask)
                if (p_loss is not None): p_losses.append(p_loss)
            # 每 n 局輸出進度提示
            if (episode % log_every == 0 or episode == num_episodes): 
                n: int = (episode - 1) % log_every + 1
                print(
                    f"已完成第 {episode} 局 | {total_steps/n:.1f} 步/局 | 截斷 {truncated} | "
                    f"示範者勝率 {target_wins/n:.0%} | "
                    f"Q-loss {sum(q_losses)/max(len(q_losses), 1):.4f} | P-loss {sum(p_losses)/max(len(p_losses), 1):.4f} | "
                    f"{(time.time()-start)/episode:.2f} 秒/局"
                )
                total_steps = truncated = target_wins = 0
                q_losses.clear(); p_losses.clear()
    
    def game_loop(self, out: bool = True, max_steps: int|None = None) -> int:
        """ 遊戲主迴圈，回傳贏家；若超過 max_steps 仍未分出勝負則回傳 -1 """
        self.deck.out = out
        for agent in self._unique_agents(): agent.eval()
        steps: int = 0
        # 主迴圈
        while True:
            if (max_steps is not None and steps >= max_steps): return -1
            steps += 1
            if (out): self.log()
            # 做出決策
            played: Card|None = self.curr_player.genmove()
            # 決定抽牌
            if (played is None): self.decide_to_draw(out=out)
            # 無效的決策
            elif (not (self.deck.is_playable(played) and self.curr_player.has(played))):
                raise ValueError("This card is not playable")
            # 正常出牌
            else: self.play_as_normal(played, out=out)
            # 判斷是否贏了
            if (self.curr_player.is_winner()): 
                if (out): self.game_over(self.turn)
                return self.turn
            # 換下一回合
            self.turn_next()

    def rank_ELO(
        self, 
        eta: float = 0.0, 
        k_val: float = 16, 
        matches: int = 100, 
        init_elo: float = 0.0, 
        networks_dir: str = "./networks", 
        table_path: str = "./ELO_table.txt",
        max_steps: int = 500
    ) -> None:
        """ 掃描 networks 資料夾底下的模型後，做成 ELO 排名表儲存到 table_path
        eta: 評估時使用 Q 網路的機率。NFSP 通常評估「平均策略」(Policy 網路)，所以預設 0
        """
        discovered_models: dict[str, ModelInfo]
        discovered_models = self.scan_networks_info(networks_dir)
        self.elo_table = ELO_table(table_path)
        # 同步更新模型資訊，並預設 ELO
        for model_name, info in discovered_models.items():
            # 現有模型
            if (model_name in self.elo_table):
                self.elo_table[model_name].p_net    = info.p_net
                self.elo_table[model_name].q_net    = info.q_net
                self.elo_table[model_name].p_memory = info.p_memory
                self.elo_table[model_name].q_memory = info.q_memory
                self.elo_table[model_name].size     = info.size
                continue
            # 新模型
            self.elo_table[model_name] = ModelInfo(
                name=info.name,
                p_net=info.p_net,
                q_net=info.q_net,
                p_memory=info.p_memory,
                q_memory=info.q_memory,
                size=info.size,
                elo=init_elo,
                matches=0
            )
        # 讓每個待測模型與 RandomAgent (ELO = 0) 對戰
        print(f"開始進行模型的對戰評分...")
        for model_name in discovered_models.keys():
            # 判斷對局數是否達標
            current_matches: int = int(self.elo_table[model_name].matches)
            games_to_play: int = matches - current_matches
            if (games_to_play <= 0): continue
            print(f"正在評估模型 {model_name}")
            info = self.elo_table[model_name]
            # 每個模型只建立、載入一次 (inference_only: 不建 trainer、不讀記憶庫)
            tested_agent: Agent = AIAgent(
                eta,
                p_network_path=info.p_net or None,
                q_network_path=info.q_net or None,
                replay_buffer_path=info.q_memory or None,
                reservior_memory_path=info.p_memory or None,
                inference_only=True,
            )
            for _ in range(games_to_play):
                ai_index: int = random.randint(0, 1)
                agents: list[Agent] = [RandomAgent(), RandomAgent()]
                agents[ai_index] = tested_agent
                try: 
                    # 重建玩家前會先歸還手牌；tested_agent 已經載入過，不會再載入
                    self._setup_players(agents)
                except RuntimeError as error:
                    print(f"跳過模型 {model_name}: {error}")
                    break
                winner: int = self.game_loop(out=False, max_steps=max_steps)
                # 計算得分與更新 ELO (超過步數上限視為和局)
                rb: float = 0.0 
                ra: float = float(self.elo_table[model_name].elo)
                expected_a: float = self.get_expected_score(ra, rb)
                score_a: float = (0.5 if winner == -1 else (1.0 if winner == ai_index else 0.0))
                # 更新模型 ELO
                self.elo_table[model_name].elo = ra + k_val*(score_a-expected_a)
                self.elo_table[model_name].matches = int(self.elo_table[model_name].matches) + 1
        # 儲存成人類可讀的形式
        self.elo_table.save()

    def test_winrate(self, num_episodes: int = 100, max_steps: int = 500) -> dict[int, float]:
        """ 對戰勝率測試 """
        winrates: dict[int, float] = {}
        num_agents: int = len(self.agents)
        if (num_agents == 0): return winrates
        for agent in self._unique_agents(): agent.eval()
        offset: int
        winner: int
        draw_count: int = 0
        current_agents: list[Agent]
        win_counts: list[int] = [0]*num_agents
        for episode in range(1, num_episodes+1):
            # 依序順序輪流換位
            offset = (episode-1) % num_agents
            current_agents = [self.agents[(i-offset)%num_agents] for i in range(num_agents)]
            try: self._setup_players(current_agents)
            except RuntimeError as error:
                print(f"設定玩家失敗: {error}")
                continue
            # 對局並記錄勝利者
            winner = self.game_loop(False, max_steps)
            # 和局/到達max_steps
            if (winner == -1):
                draw_count += 1
                continue
            # 還原贏家的真實索引
            actual_index: int = (winner+offset) % num_agents
            win_counts[actual_index] += 1
        total: float = float(num_episodes)
        draw_rate: float = float(draw_count)/total
        print(f"=== 勝率測試結果 ({num_episodes} 局) ===")
        for index, agent in enumerate(self.agents):
            winrate: float = float(win_counts[index])/total
            winrates[index] = winrate
            print(f"Agent {index} 勝率: {winrate:.1%} ({win_counts[index]} 勝)")
        print(f"平手/截斷率: {draw_rate:.1%} ({draw_count} 和)")
        return winrates

    def scan_networks_info(self, networks_dir: str = "./networks") -> dict[str, ModelInfo]: 
        """ 掃描 networks 目錄尋找各模型的 4 個 bin 檔案與計算大小 """
        discovered_models: dict[str, ModelInfo] = {}
        if (not os.path.exists(networks_dir)): return discovered_models
        for entry in os.listdir(networks_dir):
            folder_path: str = os.path.join(networks_dir, entry)
            if (not os.path.isdir(folder_path)): continue
            model_name: str = entry
            p_net: str = os.path.join(folder_path, "p_network_v1.bin")
            q_net: str = os.path.join(folder_path, "q_network_v1.bin")
            p_mem: str = os.path.join(folder_path, "ReservoirMemory_v1.bin")
            q_mem: str = os.path.join(folder_path, "ReplayBuffer_v1.bin")
            
            total_size: int = 0
            for path in [p_net, q_net, p_mem, q_mem]:
                if (os.path.exists(path)): total_size += os.path.getsize(path)

            discovered_models[model_name] = ModelInfo(
                name=model_name,
                p_net=os.path.relpath(p_net, start=".") if os.path.exists(p_net) else "",
                q_net=os.path.relpath(q_net, start=".") if os.path.exists(q_net) else "",
                p_memory=os.path.relpath(p_mem, start=".") if os.path.exists(p_mem) else "",
                q_memory=os.path.relpath(q_mem, start=".") if os.path.exists(q_mem) else "",
                size=total_size
            )
        return discovered_models

    def get_expected_score(self, rating_a: float, rating_b: float) -> float:
        """ 期望勝率計算公式 """
        return 1.0 / (1.0 + 10.0 ** ((rating_b-rating_a)/400.0))
