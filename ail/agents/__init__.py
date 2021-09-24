from ail.agents.rl_agent import PPO, SAC
from ail.agents.irl_agent import AIRL, GAIL

ALGO = {"ppo": PPO, "sac": SAC, "airl": AIRL, "gail": GAIL}
