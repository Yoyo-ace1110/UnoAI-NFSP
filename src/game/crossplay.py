import forward # type: ignore
from engine import UnoGame
from models import DQNTrainer, PolicyTrainer
from agents import Agent, RandomAgent, ClassicAgent # pyright: ignore[reportUnusedImport]

agents: list[Agent] = [ClassicAgent(), ClassicAgent()]
uno_game: UnoGame = UnoGame(agents=agents)
state_dim: int = uno_game.state_dim

q_trainer: DQNTrainer    = DQNTrainer   (state_dim)
p_trainer: PolicyTrainer = PolicyTrainer(state_dim)
q_trainer.load()
p_trainer.load()

uno_game.crossplay(
    num_episodes=16384, 
    q_trainer=q_trainer, 
    p_trainer=p_trainer,
    # 兩個ClassicAgent的經驗都拿來學
    target_player_idx=None,
    max_steps=500
)

q_trainer.save()
p_trainer.save()
