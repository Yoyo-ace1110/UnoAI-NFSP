import forward # type: ignore
from engine import UnoGame

UnoGame(agents=[]).rank_ELO(k_val=64, matches=256)
