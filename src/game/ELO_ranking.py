import forward # type: ignore
from engine import UnoGame
UnoGame(agents=[]).rank_ELO(
    k_val=32, 
    matches=16
)
