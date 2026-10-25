class Rule:
    """ 初始化規則(預設規則) """
    hand_size:            int  = 7     # 起始手牌數
    stack_skip:           bool = True  # 可否連續禁止
    stack_plus2:          bool = True  # +2 之間可連加
    stack_plus4:          bool = True  # +4 之間可連加
    stack_all_plus:       bool = False # 任何 +2, +4 都可連加
    black_finisher:       bool = False # 黑色牌能否是最後一張
    color_retention:      bool = False # 手上必須留一張顏色牌
    allow_voluntary_draw: bool = True  # 可否主動抽牌
    
    @staticmethod
    def is_valid() -> bool:
        """ 判斷規則之間是否矛盾 """
        if (Rule.stack_all_plus):
            if (not Rule.stack_plus2): return False
            if (not Rule.stack_plus4): return False
        if (Rule.black_finisher and Rule.color_retention): return False
        return True

    # TODO
    # @staticmethod
    # def load_from(path: str) -> bool: ...
    