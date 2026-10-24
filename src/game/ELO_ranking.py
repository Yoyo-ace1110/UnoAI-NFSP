import forward # type: ignore
from engine import UnoGame

UnoGame(agents=[]).rank_ELO(eta=0.85, k_val=32, matches=1024)
