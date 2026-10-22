from core import Rule, Card, Value, Deck, to_vector
from models import DQNTrainer, PolicyTrainer, print_device_info
from agents import Agent, Agent_v1, RandomAgent, HumanAgent
from .player import Player
from torch import Tensor
import torch
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
            file.write(f"{'Model Name'  :<30} | ")
            file.write(f"{'p_net'       :<30} | ")
            file.write(f"{'q_net'       :<30} | ")
            file.write(f"{'p_memory'    :<30} | ")
            file.write(f"{'q_memory'    :<30} | ")
            file.write(f"{'Size'        :<10} | ")
            file.write(f"{'ELO Rating'  :<10} | ")
            file.write(f"{'Matches'     :<8}\n")
            file.write("-" * 195 + "\n")
            for item in self.sorted():
                file.write(f"{str(item.name):<30} | ")
                file.write(f"{str(item.p_net):<30} | ")
                file.write(f"{str(item.q_net):<30} | ")
                file.write(f"{str(item.p_memory):<30} | ")
                file.write(f"{str(item.q_memory):<30} | ")
                file.write(f"{int(item.size):<10} | ")
                file.write(f"{float(item.elo):<10.2f} | ")
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

class UnoGame:
    def __init__(self, agents: list[Agent]) -> None:
        """ 初始化整個遊戲 """
        if (not Rule.is_valid()):
            raise ValueError("Invalid Rule") 
        if (len(agents)*Rule.hand_size > 108):
            raise ValueError("Too much player or initial hand size is too big")
        self.turn : int
        self.direc: int
        self.plus_sum: int
        self.deck: Deck = Deck()
        self.elo_table: ELO_table
        self.players: list[Player]
        self.agents: list[Agent] = agents
        self.players: list[Player] = []
        # 累加機制
        self.plus_sum = 0
        # 依順序輪轉
        self.turn  = 0
        self.direc = 1
        # 建立牌堆
        self.deck = Deck(self.deck.out)
        self.deck.shuffle()
        # 建立玩家並發牌
        for i in range(len(agents)):
            hand: list[Card] = self.deck.draw(Rule.hand_size)
            player: Player = Player(agents[i])
            self.players.append(player)
            player.deal_hand(hand)
            player.set_index(i)
        # 事後設定以避免輸入層的維度不同
        for agent in agents: agent.set_game(self)

    @property
    def current_player(self) -> Player: return self.players[self.turn]

    @property
    def current_agent(self) -> Agent: return self.current_player.agent

    @property
    def players_size(self) -> int: return len(self.players)

    def turn_next(self) -> None:
        """ 輪到下一個人 """
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
        self.deck.reset()
        # 重新發牌給玩家
        for player in self.players: 
            player.reset()
            player.deal_hand(self.deck.draw(Rule.hand_size))

    def game_over(self, winner_index: int) -> None:
        """ 遊戲結束 """
        print(f"Game Over! The Winner is Player{winner_index}")

    def get_tensor(self) -> Tensor:
        """ 根據目前玩家將遊戲轉為向量 """
        array: list[float] = [self.deck.size()/108]
        array.extend(to_vector(self.deck.played))
        array.extend(to_vector(self.current_player.hand))
        for player in self.players: array.append(player.hand_size()/108)
        # [牌堆張數(1)|廢牌(60)|手牌(60)|剩餘張數(玩家數量)]
        return torch.tensor(array, dtype=torch.float32)

    def decide_to_draw(self, out: bool = True) -> None:
        """ 玩家決定抽牌 """
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
            self.current_player.draw(self.deck, self.plus_sum)
            self.deck.has_effect = False # 加完之後讓牌頂失效
            self.plus_sum = 0 # 重置累加指標
            return
        # 沒牌抽
        drawed: Card|None = self.deck.draw_one()
        if (drawed is None): 
            if (out): print(f"deck is empty, Player{self.turn} drawed nothing")
            return
        # 抽一張
        self.current_player.receive(drawed)
        if (isinstance(self.current_player.agent, HumanAgent)):
            if (out): print(f"Player{self.turn} drawed")
            drawed.cout(False)
            return
        if (out): print(f"Player{self.turn} drawed 1 card")

    def play_as_normal(self, played: Card, out: bool = True) -> None:
        """ 正常出牌 """
        self.deck.set_top(played)
        self.current_player.remove(played)
        if (out): print(f"Player{self.turn} played", end="")
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
        if (self.current_player.uno()): 
            if (out): print(f"Player{self.turn} Uno!")

    def selfplay(self, num_episodes: int) -> None:
        """ 模型自我對戰訓練 """
        self.deck.out = False   # 避免不必要的輸出
        print_device_info()     # 輸出一次使用裝置
        for agent in self.agents: agent.train()
        print(f"開始進行 {num_episodes} 局自我對戰")
        for episode in range(1, num_episodes+1):
            self.reset()
            while True:
                state = self.get_tensor()
                # 執行動作
                played_card: Card|None = self.current_player.genmove(self.deck.top, self.deck.has_effect)
                # 決定抽牌
                if (played_card is None): self.decide_to_draw(out=False)
                # 無效的決策
                elif (not self.deck.is_playable(played_card)):
                    raise ValueError("Bot played the Card that is not playable")
                # 正常出牌
                else: self.play_as_normal(played_card, out=False)
                next_state: Tensor = self.get_tensor()
                # 檢查是否遊戲結束
                done: bool = self.current_player.is_winner()
                if (done): reward: float = 1.0
                elif played_card is not None:
                    reward: float = 0.04
                else: reward: float = -0.04
                # 存入雙記憶庫(Dual Memory Mechanism)並更新雙網路
                self.current_agent.push_q(state, played_card, reward, next_state, done)
                self.current_agent.push_p(state, played_card)
                # 延後中斷遊戲(先訓練)
                if (done): break
                self.turn_next()
            # 每 n 局輸出進度提示
            if (episode%16 == 0): print(f"已完成第 {episode} 局")

    def crossplay(self, num_episodes: int, q_trainer: DQNTrainer, p_trainer: PolicyTrainer) -> None:
        """ 模型交叉對戰訓練 """
        self.deck.out = False   # 避免不必要的輸出
        print_device_info()     # 輸出一次使用裝置
        for agent in self.agents: agent.eval()
        print(f"開始進行 {num_episodes} 局交叉對戰")
        for episode in range(1, num_episodes+1):
            self.reset()
            while True:
                state = self.get_tensor()
                # 執行動作並推進遊戲環境
                played_card: Card|None = self.current_player.genmove(self.deck.top, self.deck.has_effect)
                # 決定抽牌
                if (played_card is None): self.decide_to_draw(out=False)
                # 無效的決策
                elif (not self.deck.is_playable(played_card)):
                    raise ValueError("Bot played the Card that is not playable")
                # 正常出牌
                else: self.play_as_normal(played_card, out=False)
                next_state: Tensor = self.get_tensor()
                # 檢查是否遊戲結束
                done: bool = self.current_player.is_winner()
                if (done): reward: float = 1.0
                elif played_card is not None:
                    reward: float = 0.04
                else: reward: float = -0.04
                # 存入雙記憶庫(Dual Memory Mechanism)並更新雙網路 
                q_trainer.push(state, played_card, reward, next_state, done)
                p_trainer.push(state, played_card)
                # 延後中斷遊戲(先訓練)
                if (done): break
                self.turn_next()
            # 每 n 局輸出進度提示
            if (episode%16 == 0): print(f"已完成第 {episode} 局")

    def game_loop(self, out: bool = True) -> int:
        """ 遊戲主迴圈，回傳贏家 """
        self.deck.out = out
        for agent in self.agents: agent.eval()
        # 主迴圈
        while True:
            self.log()
            # 做出決策
            played: Card|None = self.current_player.genmove(self.deck.top, self.deck.has_effect)
            # 決定抽牌
            if (played is None): self.decide_to_draw(out=out)
            # 無效的決策
            elif (not self.deck.is_playable(played)):
                # 人類打錯牌
                if (isinstance(self.current_player.agent, HumanAgent)):
                    if (out): print("The card is not playable, please try again")
                    continue
                # AI 出錯
                raise ValueError("Bot played the Card that is not playable")
            # 正常出牌
            else: self.play_as_normal(played, out=out)
            # 判斷是否贏了
            if (self.current_player.is_winner()): 
                self.game_over(self.turn)
                return self.turn
            # 換下一回合
            self.turn_next()
    
    def rank_ELO(
        self, 
        eta: float = 0.85, 
        k_val: float = 16, 
        matches: int = 100, 
        init_elo: float = 0.0, 
        networks_dir: str = "./networks", 
        table_path: str = "./ELO_table.txt"
    ) -> None:
        """ 掃描 networks 資料夾底下的模型後，做成 ELO 排名表儲存到 table_path """
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
            for _ in range(games_to_play):
                # 建立待測模型與 RandomAgent
                tested_agent: Agent = Agent_v1(
                    eta,
                    p_network_path=info.p_net or None,
                    q_network_path=info.q_net or None,
                    replay_buffer_path=info.q_memory or None,
                    reservior_memory_path=info.p_memory or None,
                )
                self.agents = [tested_agent, RandomAgent()]
                self.players = []
                self.deck.reset()
                for i, agent in enumerate(self.agents):
                    hand: list[Card] = self.deck.draw(Rule.hand_size)
                    player: Player = Player(agent)
                    self.players.append(player)
                    player.deal_hand(hand)
                    player.set_index(i)
                for agent in self.agents: agent.set_game(self)
                self.turn  = 0
                self.direc = 1
                self.plus_sum = 0
                winner: int = self.game_loop(out=False)
                # 計算得分與更新 ELO
                rb: float = 0.0 
                ra: float = float(self.elo_table[model_name].elo)
                expected_a: float = self.get_expected_score(ra, rb)
                score_a: float = (1.0 if winner == 0 else 0.0)
                # 更新模型 ELO
                self.elo_table[model_name].elo = ra + k_val*(score_a-expected_a)
                self.elo_table[model_name].matches = int(self.elo_table[model_name].matches) + 1
        # 儲存成人類可讀的形式
        self.elo_table.save()

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
