from torch import Tensor
from game import UnoGame
from agents import Agent, RandomAgent
from models import DQNTrainer, PolicyTrainer

agents: list[Agent] = [RandomAgent(), RandomAgent()]
uno_game: UnoGame = UnoGame(agents=agents, eta=0.8)
sample_tensor: Tensor = uno_game.get_tensor()
state_dim: int = sample_tensor.shape[0]

q_trainer: DQNTrainer    = DQNTrainer   (state_dim)
p_trainer: PolicyTrainer = PolicyTrainer(state_dim)
q_trainer.load()
p_trainer.load()

uno_game.train_loop(
    num_episodes=1000, 
    q_trainer=q_trainer, 
    p_trainer=p_trainer
)

q_trainer.save()
p_trainer.save()
