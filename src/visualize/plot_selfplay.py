# type: ignore
from __future__ import annotations
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from log_parser import parse_selfplay_file, SelfPlayData

def main() -> None:
    """主程式: 繪製自我對戰圖表"""
    data: SelfPlayData = parse_selfplay_file("src\\visualize\\selfplay.log")
    if not data.episodes:
        print("無自我對戰資料可供繪製")
        return

    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei', 'SimHei', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False

    fig: Figure
    axs: list[list[Axes]]
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Self-play 訓練日誌分析趨勢圖", fontsize=16, fontweight='bold')

    ax_step: Axes = axs[0, 0]
    ax_step.plot(data.episodes, data.steps, color='royalblue')
    ax_step.set_title("平均每局步數")
    ax_step.set_xlabel("局數")
    ax_step.set_ylabel("步數")
    ax_step.grid(True, linestyle='--', alpha=0.6)

    ax_loss: Axes = axs[0, 1]
    ax_loss.plot(data.episodes, data.q_losses, label='Q-loss', color='royalblue')
    ax_loss.plot(data.episodes, data.p_losses, label='P-loss', color='cornflowerblue', linestyle='--')
    ax_loss.set_title("網路 Loss 變化")
    ax_loss.set_xlabel("局數")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend()
    ax_loss.grid(True, linestyle='--', alpha=0.6)

    ax_win: Axes = axs[1, 0]
    ax_win.plot(data.episodes, data.win_rates, label='0號位勝率', color='forestgreen')
    ax_win.axhline(50, color='gray', linestyle=':', label='均衡線')
    ax_win.set_title("0號位勝率")
    ax_win.set_xlabel("局數")
    ax_win.set_ylabel("勝率 (%)")
    ax_win.set_ylim(0, 100)
    ax_win.legend()
    ax_win.grid(True, linestyle='--', alpha=0.6)

    ax_draw: Axes = axs[1, 1]
    ax_draw.plot(data.episodes, data.draw_rates, label='抽牌率', color='darkorange')
    ax_draw.set_title("抽牌率趨勢")
    ax_draw.set_xlabel("局數")
    ax_draw.set_ylabel("抽牌率 (%)")
    ax_draw.legend()
    ax_draw.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("selfplay_result.png", dpi=300)
    print("圖表已成功儲存至 selfplay_result.png")

if __name__ == "__main__": main()
