from game import UnoGame
from agents import Agent, Agent_v1, HumanAgent, RandomAgent # pyright: ignore[reportUnusedImport]

agents: list[Agent] = [HumanAgent(), Agent_v1()]
uno_game: UnoGame = UnoGame(agents=agents)
uno_game.game_loop()
