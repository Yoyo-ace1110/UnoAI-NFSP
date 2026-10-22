import forward # type: ignore
from engine import UnoGame
from agents import Agent, Agent_v1, HumanAgent # pyright: ignore[reportUnusedImport]

agents: list[Agent] = [Agent_v1(), HumanAgent()]
uno_game: UnoGame = UnoGame(agents=agents)
uno_game.game_loop()
