import forward # type: ignore
from engine import UnoGame

UnoGame(agents=[]).rank_ELO(
    eta=0.0, 
    k_val=64,
    matches=256, 
    networks_dir="./networks/2P_rule2"
)
