import forward # type: ignore
from engine import UnoGame

# eta=0: 評估 NFSP 的平均策略 (Policy 網路)；想比較最佳回應(Q 網路)可改成 eta=1
UnoGame(agents=[]).rank_ELO(eta=0.0, k_val=64, matches=256)
