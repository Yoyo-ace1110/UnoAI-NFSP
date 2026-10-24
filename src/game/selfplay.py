import forward # type: ignore
from engine import UnoGame
from agents import Agent, AIAgent

agent: AIAgent = AIAgent()
agents: list[Agent] = [agent, agent]
uno_game: UnoGame = UnoGame(agents=agents)
uno_game.selfplay(num_episodes=16384, max_steps=500, log_every=16, save_every=256)
agent.save()
