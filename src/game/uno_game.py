from core import Rule, Card, Value, Deck, to_vector
from models import DQNTrainer, PolicyTrainer, print_device_info
from agents import Agent, HumanAgent
from .player import Player
from torch import Tensor
import torch

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
        self.players: list[Player]
        self.agents: list[Agent] = agents
        self.reset(agents=self.agents)

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

    def reset(self, agents: list[Agent]) -> None:
        """ 重製遊戲狀態 """
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

    def train_loop(self, num_episodes: int, q_trainer: DQNTrainer, p_trainer: PolicyTrainer) -> None:
        """ 訓練主迴圈：執行多局自我對戰並進行 NFSP 雙網路訓練 """
        self.deck.out = False   # 避免不必要的輸出
        print_device_info()     # 輸出一次使用裝置
        print(f"開始進行 {num_episodes} 局自我對戰")
        for episode in range(1, num_episodes+1):
            self.reset(agents=self.agents)
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
            if (episode%4 == 0): print(f"已完成第 {episode} 局自我對戰")

    def game_loop(self) -> None:
        """ 遊戲主迴圈 """
        self.deck.out = True
        while True:
            self.log()
            # 做出決策
            played: Card|None = self.current_player.genmove(self.deck.top, self.deck.has_effect)
            # 決定抽牌
            if (played is None): self.decide_to_draw()
            # 無效的決策
            elif (not self.deck.is_playable(played)):
                # 人類打錯牌
                if (isinstance(self.current_player.agent, HumanAgent)):
                    print("The card is not playable, please try again")
                    continue
                # AI 出錯
                raise ValueError("Bot played the Card that is not playable")
            # 正常出牌
            else: self.play_as_normal(played)
            # 判斷是否贏了
            if (self.current_player.is_winner()): 
                self.game_over(self.turn)
                break
            # 換下一回合
            self.turn_next()
