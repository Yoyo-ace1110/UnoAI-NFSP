from game import UnoGame
from models import QNetwork, PNetwork
from agents import Agent, HumanAgent, RandomAgent

agents: list[Agent] = [HumanAgent(), RandomAgent()]
uno_game: UnoGame = UnoGame(agents=agents, eta=0.8)
uno_game.train_loop(
    num_episodes=10, 
    q_buffer=, 
    p_memory=
)
