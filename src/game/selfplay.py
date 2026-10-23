import forward # type: ignore
from engine import UnoGame
from agents import Agent, AIAgent

agent: AIAgent = AIAgent()
agents: list[Agent] = [agent, agent]
uno_game: UnoGame = UnoGame(agents=agents)
uno_game.selfplay(num_episodes=1024)
agent.save()
