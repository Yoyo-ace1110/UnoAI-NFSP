from __future__ import annotations
import re

class SelfPlayData:
    def __init__(self) -> None:
        self.episodes:      list[ int ] = []
        self.steps:         list[float] = []
        self.draw_rates:    list[float] = []
        self.win_rates:     list[float] = []
        self.q_losses:      list[float] = []
        self.p_losses:      list[float] = []
        self.speeds:        list[float] = []

class CrossPlayData:
    def __init__(self) -> None:
        self.episodes:      list[ int ] = []
        self.steps:         list[float] = []
        self.win_rates:     list[float] = []
        self.q_losses:      list[float] = []
        self.p_losses:      list[float] = []
        self.speeds:        list[float] = []

def smooth_data(values: list[float], window: int = 15) -> list[float]:
    """計算指定視窗大小的移動平均，使圖表平滑"""
    smoothed: list[float] = []
    n: int = len(values)
    half_w: int = window // 2
    i: int
    for i in range(n):
        start: int = max(0, i - half_w)
        end: int = min(n, i + half_w + 1)
        chunk: list[float] = values[start : end]
        avg: float = sum(chunk) / float(len(chunk))
        smoothed.append(avg)
    return smoothed

def parse_selfplay_file(filepath: str) -> SelfPlayData:
    """讀取並解析自我對戰日誌"""
    data: SelfPlayData = SelfPlayData()
    pattern: re.Pattern[str] = re.compile(
        r"已完成第\s*(\d+)\s*局\s*\|\s*"
        r"([\d.]+)\s*步/局\s*\|\s*"
        r"抽牌率\s*([\d.]+)%\s*\|\s*"
        r"截斷\s*\d+\s*\|\s*"
        r"0號位勝率\s*([\d.]+)%\s*\|\s*"
        r"Q-loss\s*([\d.]+)\s*\|\s*"
        r"P-loss\s*([\d.]+)\s*\|\s*"
        r"([\d.]+)\s*秒/局"
    )
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                match = pattern.search(line)
                if (not match): continue
                data.episodes.  append( int (match.group(1)))
                data.steps.     append(float(match.group(2)))
                data.draw_rates.append(float(match.group(3)))
                data.win_rates. append(float(match.group(4)))
                data.q_losses.  append(float(match.group(5)))
                data.p_losses.  append(float(match.group(6)))
                data.speeds.    append(float(match.group(7)))
    except FileNotFoundError: print(f"找不到檔案: {filepath}")
    
    if (data.steps):
        data.steps      = smooth_data(data.steps, 32)
        data.draw_rates = smooth_data(data.draw_rates, 32)
        data.win_rates  = smooth_data(data.win_rates, 20)
        data.q_losses   = smooth_data(data.q_losses, 15)
        data.p_losses   = smooth_data(data.p_losses, 15)
        data.speeds     = data.speeds
    return data

def parse_crossplay_file(filepath: str) -> CrossPlayData:
    """讀取並解析交叉對戰日誌"""
    data: CrossPlayData = CrossPlayData()
    pattern: re.Pattern[str] = re.compile(
        r"已完成第\s*(\d+)\s*局\s*\|\s*"
        r"([\d.]+)\s*步/局\s*\|\s*"
        r"截斷\s*\d+\s*\|\s*"
        r"示範者勝率\s*([\d.]+)%\s*\|\s*"
        r"Q-loss\s*([\d.]+)\s*\|\s*"
        r"P-loss\s*([\d.]+)\s*\|\s*"
        r"([\d.]+)\s*秒/局"
    )
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                match = pattern.search(line)
                if (not match): continue
                data.episodes. append( int (match.group(1)))
                data.steps.    append(float(match.group(2)))
                data.win_rates.append(float(match.group(3)))
                data.q_losses. append(float(match.group(4)))
                data.p_losses. append(float(match.group(5)))
                data.speeds.   append(float(match.group(6)))
    except FileNotFoundError: print(f"找不到檔案: {filepath}")
    
    if (data.steps):
        data.steps      = smooth_data(data.steps    , 32)
        data.win_rates  = smooth_data(data.win_rates, 20)
        data.q_losses   = data.q_losses
        data.p_losses   = data.p_losses
        data.speeds     = data.speeds
    return data
