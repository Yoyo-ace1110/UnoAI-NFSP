import forward # type: ignore
from engine import UnoGame
from agents import * # pyright: ignore[reportUnusedImport]

"""
selfplayed_path: str = "networks\\2P_rule1\\2_selfplay32768+\\"
selfplayed_AI: AIAgent = AIAgent(
    eta=0.0, 
    p_network_path=       f"{selfplayed_path}p_network_v1.bin", 
    q_network_path=       f"{selfplayed_path}q_network_v1.bin", 
    replay_buffer_path=   f"{selfplayed_path}ReplayBuffer_v1.bin", 
    reservior_memory_path=f"{selfplayed_path}ReservoirMemory_v1.bin", 
)
"""

crossplayed_path: str = "networks\\2P_rule2\\1_crossplay512+\\"
crossplayed_AI: AIAgent = AIAgent(
    eta=0.0, 
    p_network_path=       f"{crossplayed_path}p_network_v1.bin", 
    q_network_path=       f"{crossplayed_path}q_network_v1.bin", 
    replay_buffer_path=   f"{crossplayed_path}ReplayBuffer_v1.bin", 
    reservior_memory_path=f"{crossplayed_path}ReservoirMemory_v1.bin", 
)

agents: list[Agent] = [crossplayed_AI, crossplayed_AI]
uno_game: UnoGame = UnoGame(agents=agents)
uno_game.test_winrate(num_episodes=1000)
