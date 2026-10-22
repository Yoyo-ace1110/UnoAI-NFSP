import forward # type: ignore
from engine import UnoGame
UnoGame(agents=[]).rank_ELO(
    k_val=128, 
    matches=8, 
    init_elo=0
)
