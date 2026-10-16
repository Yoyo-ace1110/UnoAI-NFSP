from game import UnoGame
from agents import Agent, HumanAgent, RandomAgent

agents: list[Agent] = [HumanAgent(), RandomAgent()]
uno_game: UnoGame = UnoGame(agents=agents, eta=0.8)
uno_game.game_loop()
