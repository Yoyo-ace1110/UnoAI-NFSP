import forward # type: ignore
from engine import UnoGame
from agents import Agent, AIAgent, HumanAgent # pyright: ignore[reportUnusedImport]

agents: list[Agent] = [AIAgent(), HumanAgent()]
uno_game: UnoGame = UnoGame(agents=agents)
uno_game.game_loop()
