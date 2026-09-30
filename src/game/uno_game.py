from core import Rule, Card, Deck
from core import Card, Value
from player import Player
from agents import Agent

class UnoGame:
    def __init__(self, agents: list[Agent]) -> None:
        """ 初始化整個遊戲 """
        if (not Rule.is_valid()):               raise ValueError("Invalid Rule") 
        if (len(agents)*Rule.hand_size > 108):  raise ValueError("Too much player or initial hand size is too big")
        # 依順序輪轉
        self.turn: int = 0
        self.direc: int = 1
        # 建立牌堆
        self.deck: Deck = Deck()
        self.deck.shuffle()
        # 建立玩家並發牌
        self.players: list[Player] = []
        for i in range(len(agents)):
            hand: list[Card] = self.deck.draw(Rule.hand_size)
            player: Player = Player(agents[i])
            player.deal_hand(hand)
            self.players.append(player)
        # 累加機制
        self.plus_sum: int = 0
        # 效果處理
        self.has_effect: bool = False

    @property
    def current_player(self) -> Player: return self.players[self.turn]

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
        print(f"top card: {self.deck.top}")
        for i in range(self.players_size):
            # 當前玩加輸出手牌, 其餘輸出張數
            if (self.turn == i): self.players[i].show_hand(i)
            else: print(f"Player{i}: {self.players[i].hand_size()} cards")

    def game_over(self, winner_index: int) -> None:
        """ 遊戲結束 """
        print(f"Game Over! The Winner is Player{winner_index}")

    def play_as_normal(self, played: Card) -> None:
        """ 正常出牌 """
        self.deck.set_top(played)
        # 迴轉機制
        if (played.value == Value.turn):
            self.reverse()
        # 禁止機制
        elif (played.value == Value.skip):
            self.has_effect = True
        # 連加機制
        elif (played.value == Value.plus2):
            self.has_effect = True
            self.plus_sum += 2
        elif (played.value == Value.plus4):
            self.has_effect = True
            self.plus_sum += 4
        # Uno 機制
        if (self.current_player.uno()): 
            print(f"Player{self.turn} Uno!")

    def game_loop(self) -> None:
        """ 遊戲主迴圈 """
        while True:
            self.log()
            # 做出決策
            played: Card|None = self.current_player.genmove(self.deck.top)
            # 決定抽牌
            if (played is None):
                top_is_skip : bool = (self.deck.top.value == Value.skip  and self.has_effect)
                top_is_plus2: bool = (self.deck.top.value == Value.plus2 and self.has_effect)
                top_is_plus4: bool = (self.deck.top.value == Value.plus4 and self.has_effect)
                # 被禁止不用抽
                if (top_is_skip): 
                    self.has_effect = False
                # 被加牌
                elif (top_is_plus2 or top_is_plus4): 
                    self.current_player.draw(self.deck, self.plus_sum)
                    self.has_effect = False # 加完之後讓牌頂失效
                    self.plus_sum = 0 # 重置累加指標
                # 單純抽牌
                else:
                    # 抽一張
                    if (Rule.draw_one):
                        drawed: Card|None = self.deck.draw_one()
                        # 沒牌抽
                        if (drawed is None): pass
                        # 打掉這張
                        elif (self.current_player.play_drawed(drawed)):
                            self.play_as_normal(drawed)
                        # 收下這張
                        else: self.current_player.receive(drawed) 
                    # 抽到可以打
                    else: self.current_player.draw_to_play(self.deck)
            # 無效的決策
            elif (not self.deck.is_playable(played)): 
                raise ValueError("The Card is not playable")
            # 正常出牌
            else: self.play_as_normal(played)
            # 判斷是否贏了
            if (self.current_player.is_winner()): 
                self.game_over(winner_index=self.turn)
                break   
            # 判斷是否持續洗牌
            if (Rule.always_suffle): self.deck.reshuffle()
            # 換下一回合
            self.turn_next()

    def train_loop(self) -> None:
        """ 訓練主迴圈 """
