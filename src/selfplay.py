from game import UnoGame
from agents import Agent, Agent_v1

agent: Agent_v1 = Agent_v1()
agents: list[Agent] = [agent, agent]
uno_game: UnoGame = UnoGame(agents=agents)
uno_game.selfplay(num_episodes=16384)
agent.save()
