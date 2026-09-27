class Agent: ...

class Rule:
    """ 初始化規則(預設規則) """
    hand_size:      int  = 7     # 一開始發幾張牌
    stack_skip:     bool = True  # 可否連續禁止
    stack_plus2:    bool = True  # +2 之間可連加
    stack_plus4:    bool = True  # +4 之間可連加
    stack_all_plus: bool = True  # 任何 +2, +4 都可連加
    draw_one:       bool = True  # 沒牌時僅抽一張
    draw_to_play:   bool = False # 沒牌時持續抽牌直到能出牌
    play_after_draw:bool = True  # 抽牌之後是否可以立即打掉
    black_finisher: bool = False # 黑色牌能否是最後一張
    color_retention:bool = True  # 手上必須留一張顏色牌
    always_suffle:  bool = False # 每打一張牌就洗進牌堆(只留當前的牌)
    
    # TODO
    # @staticmethod
    # def load_from(path: str) -> bool: ...
    
    @staticmethod
    def is_valid() -> bool:
        """ 判斷規則之間是否矛盾 """
        if (Rule.draw_to_play):
            if (Rule.draw_one):            return False
            if (not Rule.play_after_draw): return False
        if (Rule.stack_all_plus):
            if (not Rule.stack_plus2): return False
            if (not Rule.stack_plus4): return False
        if (Rule.black_finisher and Rule.color_retention): return False
        return True
