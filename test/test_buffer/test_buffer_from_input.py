import pytest

import gym
import numpy as np
import torch as th

from ail.buffer import RolloutBuffer

env = gym.make("Pendulum-v0")

state, done = env.reset(), False
action = env.action_space.sample()
next_state, reward, done, info = env.step(action)

data = dict(
    obs=state.astype(np.float32).reshape(1, -1),
    acts=action.astype(np.float32).reshape(1, -1),
    rews=reward.astype(np.float32).reshape(1, -1),
    dones=np.asarray(done, dtype=np.float32).reshape(1, -1),
    next_obs=next_state.astype(np.float32).reshape(1, -1),
)


def test_buffer_init_with_env():
    try:
        buffer = RolloutBuffer(
            capacity=1_000,
            device="cpu",
            env=env,
        )
    except AssertionError:
        with pytest.raises(AssertionError) as exc_info:
            assert "Env must be an instance of GymEnv." in str(exc_info.value)


def test_buffer_init_with_inputs():
    try:
        buffer = RolloutBuffer(
            capacity=1_000,
            device="cpu",
            obs_shape=env.observation_space.shape,
            obs_dtype=env.observation_space.dtype,
            act_shape=env.action_space.shape,
            act_dtype=env.action_space.dtype,
        )
    except AssertionError as exc_info:
        assert "Runtime Error" in str(exc_info.value)


def test_buffer_init_size():
    buffer = RolloutBuffer(
        capacity=1_000,
        device="cpu",
        obs_shape=env.observation_space.shape,
        obs_dtype=env.observation_space.dtype,
        act_shape=env.action_space.shape,
        act_dtype=env.action_space.dtype,
    )
    assert buffer.size() == 0
    assert isinstance(buffer.size(), int)


def test_buffer_post_size():
    buffer = RolloutBuffer(
        capacity=1_000,
        device="cpu",
        obs_shape=env.observation_space.shape,
        obs_dtype=env.observation_space.dtype,
        act_shape=env.action_space.shape,
        act_dtype=env.action_space.dtype,
    )
    for i in range(1, 100):
        buffer.store(data)
        assert isinstance(buffer.size(), int)
        assert buffer.size() == i


def test_buffer_max_size_with_truncate_ok():
    buffer = RolloutBuffer(
        capacity=1_000,
        device="cpu",
        obs_shape=env.observation_space.shape,
        obs_dtype=env.observation_space.dtype,
        act_shape=env.action_space.shape,
        act_dtype=env.action_space.dtype,
    )
    for i in range(1, 2_000):
        buffer.store(data, truncate_ok=True)
        assert isinstance(buffer.size(), int)
        if i <= 1000:
            assert buffer.size() == i
        else:
            assert buffer.size() == 1000


def test_buffer_max_size_without_truncate_ok():
    buffer = RolloutBuffer(
        capacity=1_000,
        device="cpu",
        obs_shape=env.observation_space.shape,
        obs_dtype=env.observation_space.dtype,
        act_shape=env.action_space.shape,
        act_dtype=env.action_space.dtype,
    )
    for i in range(1, 1_000):
        buffer.store(data, truncate_ok=False)
        assert isinstance(buffer.size(), int)
        if i <= 1000:
            assert buffer.size() == i
    try:
        buffer.store(data, truncate_ok=False)
    except AssertionError as exc_info:
        with pytest.raises(ValueError) as exc_info:
            assert "Not enough capacity to store data." in str(exc_info.value)


def test_buffer_sample_size_type_dtype():
    n_samples = 166
    buffer = RolloutBuffer(
        capacity=1_000,
        device="cpu",
        obs_shape=env.observation_space.shape,
        obs_dtype=env.observation_space.dtype,
        act_shape=env.action_space.shape,
        act_dtype=env.action_space.dtype,
    )
    for _ in range(1, 1_000):
        buffer.store(data, truncate_ok=False)

    samples = buffer.sample(n_samples)
    assert isinstance(samples, dict)
    for k, v in samples.items():
        assert isinstance(v, th.Tensor)
        assert v.shape[0] == n_samples
        assert v.dtype == th.float32


def test_buffer_store_extra_keys():
    buffer = RolloutBuffer(
        capacity=1_000,
        device="cpu",
        obs_shape=env.observation_space.shape,
        obs_dtype=env.observation_space.dtype,
        act_shape=env.action_space.shape,
        act_dtype=env.action_space.dtype,
        extra_shapes={"advs": (1,), "log_pis": (1,)},
        extra_dtypes={"advs": np.float32, "log_pis": np.float32},
    )
    data.update({"advs": np.zeros(1), "log_pis": np.ones(1)})
    for _ in range(1, 1_000):
        buffer.store(data, truncate_ok=False)

    n_samples = 16
    samples = buffer.sample(n_samples)
    assert isinstance(samples, dict)

    assert "advs" in samples.keys()
    assert "log_pis" in samples.keys()
    assert len(samples.keys()) == 7
    # samples.keys():
    # (['obs', 'acts', 'next_obs', 'dones', 'rews', 'adv', 'log_pis'])
    for k, v in samples.items():
        assert isinstance(v, th.Tensor)
        assert v.shape[0] == n_samples
        assert v.dtype == th.float32
        if k not in ["obs", "acts", "next_obs"]:
            assert v.shape[1] == 1


def test_buffer_store_paths_without_truncate():
    env = gym.make("Pendulum-v0")

    buffer = RolloutBuffer(
        capacity=1_000,
        device="cpu",
        obs_shape=env.observation_space.shape,
        obs_dtype=env.observation_space.dtype,
        act_shape=env.action_space.shape,
        act_dtype=env.action_space.dtype,
    )

    state, done = env.reset(), False
    obs, act, next_obs, rew = [], [], [], []
    for _ in range(2_000):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)

        obs += [state]
        act += [action]
        next_obs += [next_state]
        rew += [reward]

        state = next_state
        if done:
            state, done = env.reset(), False

    data = dict(
        obs=np.asarray(obs, dtype=np.float32),
        acts=np.asarray(act, dtype=np.float32),
        rews=np.asarray(rew, dtype=np.float32).reshape(-1, 1),
        next_obs=np.asarray(next_obs, dtype=np.float32),
    )
    try:
        buffer.store_path(transitions=data, truncate_ok=False)
    except ValueError as exc_info:
        with pytest.raises(AssertionError) as exc_info:
            assert "Not enough capacity to store data." in str(exc_info.value)


def test_buffer_store_paths_with_truncate():
    env = gym.make("Pendulum-v0")

    buffer = RolloutBuffer(
        capacity=1_000,
        device="cpu",
        obs_shape=env.observation_space.shape,
        obs_dtype=env.observation_space.dtype,
        act_shape=env.action_space.shape,
        act_dtype=env.action_space.dtype,
    )

    state, done = env.reset(), False
    obs, act, next_obs, rew = [], [], [], []
    for _ in range(2_000):
        action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)

        obs += [state]
        act += [action]
        next_obs += [next_state]
        rew += [reward]

        state = next_state
        if done:
            state, done = env.reset(), False

    data = dict(
        obs=np.asarray(obs, dtype=np.float32),
        acts=np.asarray(act, dtype=np.float32),
        rews=np.asarray(rew, dtype=np.float32).reshape(-1, 1),
        next_obs=np.asarray(next_obs, dtype=np.float32),
    )
    buffer.store_path(transitions=data, truncate_ok=True)
    assert buffer.size() == 1_000


def test_buffer_get_all():
    buffer = RolloutBuffer(
        capacity=1_000,
        device="cpu",
        obs_shape=env.observation_space.shape,
        obs_dtype=env.observation_space.dtype,
        act_shape=env.action_space.shape,
        act_dtype=env.action_space.dtype,
    )

    for _ in range(0, 1_000):
        buffer.store(data, truncate_ok=False)

    samples = buffer.get()
    assert isinstance(samples, dict)

    for k, v in samples.items():
        assert isinstance(v, th.Tensor)
        assert v.shape[0] == 1_000
        assert v.dtype == th.float32
        if k not in ["obs", "acts", "next_obs"]:
            assert v.shape[1] == 1


def test_buffer_get_sample():
    buffer = RolloutBuffer(
        capacity=1_000,
        device="cpu",
        obs_shape=env.observation_space.shape,
        obs_dtype=env.observation_space.dtype,
        act_shape=env.action_space.shape,
        act_dtype=env.action_space.dtype,
    )

    for _ in range(0, 1_000):
        buffer.store(data, truncate_ok=False)

    n_samples = 500
    samples = buffer.get(n_samples)
    assert isinstance(samples, dict)

    for k, v in samples.items():
        assert isinstance(v, th.Tensor)
        assert v.shape[0] == 500
        assert v.dtype == th.float32
        if k not in ["obs", "acts", "next_obs"]:
            assert v.shape[1] == 1
