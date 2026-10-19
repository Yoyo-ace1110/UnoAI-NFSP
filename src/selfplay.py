from game import UnoGame
from agents import Agent_v1

uno_game: UnoGame = UnoGame(agents=[Agent_v1(), Agent_v1()])
uno_game.selfplay(num_episodes=1024)
