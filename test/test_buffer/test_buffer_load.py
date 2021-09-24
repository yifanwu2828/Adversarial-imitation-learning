import pathlib

import gym
import numpy as np
import torch as th

from pympler import asizeof

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

from ail.buffer import RolloutBuffer, ReplayBuffer, BufferTag

env_id = "HalfCheetah-v2"
env = gym.make(env_id)

print(env.observation_space)
print(env.action_space)


state, done = env.reset(), False
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)

# Path
path = pathlib.Path(__file__).parent.resolve()
print(f"current_dir: {path}")

data_path = path.parent.parent / "scripts" / "transitions" / env_id / "size11000.npz"
data = dict(np.load(data_path))
ic(data)
buffer = ReplayBuffer.from_data(data, device="cpu", with_reward=False)


trajectory = buffer.get()
for k, v in trajectory.items():
    print(k, v.shape)
    assert v.shape[0] == 11000
print("\n")

samples = buffer.sample(10)
for k, v in samples.items():
    print(k, v.shape)
    assert v.shape[0] == 10

print(asizeof.asizeof(buffer))

a = BufferTag.REPLAY

print(a == buffer.tag)


def normalize_obs():
    pass