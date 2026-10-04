from game import UnoGame
from agents import Agent, HumanAgent, RandomAgent

agents: list[Agent] = [HumanAgent(), RandomAgent()]
uno_game: UnoGame = UnoGame(agents=agents)
uno_game.game_loop()

# TODO: hand sorting
# human agent not playable move => try again
