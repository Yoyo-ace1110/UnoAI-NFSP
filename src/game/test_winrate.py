import forward # type: ignore
from engine import UnoGame
from agents import * # pyright: ignore[reportUnusedImport]

selfplayed_path: str = "networks\\3_selfplay_only_32768\\"
selfplayed_AI: AIAgent = AIAgent(
    eta=1.0, 
    p_network_path=       f"{selfplayed_path}p_network_v1.bin", 
    q_network_path=       f"{selfplayed_path}q_network_v1.bin", 
    replay_buffer_path=   f"{selfplayed_path}ReplayBuffer_v1.bin", 
    reservior_memory_path=f"{selfplayed_path}ReservoirMemory_v1.bin", 
)
mix_played_path: str = "networks\\2_selfplay_after_16384+\\"
mix_played_AI: AIAgent = AIAgent(
    eta=1.0, 
    p_network_path=       f"{mix_played_path}p_network_v1.bin", 
    q_network_path=       f"{mix_played_path}q_network_v1.bin", 
    replay_buffer_path=   f"{mix_played_path}ReplayBuffer_v1.bin", 
    reservior_memory_path=f"{mix_played_path}ReservoirMemory_v1.bin", 
)

agents: list[Agent] = [RandomAgent(), selfplayed_AI]
uno_game: UnoGame = UnoGame(agents=agents)
uno_game.test_winrate(num_episodes=1000)
