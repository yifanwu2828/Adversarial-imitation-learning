from gym.envs.registration import register

register(
    id="NavEnv-v0",
    entry_point="gym_nav.envs:NavEnv",
)

register(
    id="MultiNavEnv-v0",
    entry_point="gym_nav.envs:MultiNavEnv",
)
