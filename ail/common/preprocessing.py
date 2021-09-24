import numpy as np

from ail.common.math import normalize


def make_absorbing_states(obs: np.ndarray, dones: np.ndarray) -> np.ndarray:
    """
    Absorbing states
    A technique introduced in the DAC (https://arxiv.org/pdf/1809.02925.pdf)
    to mitigate the bias and encourage the policy to generate episodes of similar
    length to demonstrations.
    """
    combined_states = np.hstack([obs, dones])
    absorbing_states = np.zeros(combined_states.shape[1]).reshape(1, -1)
    absorbing_states[:, -1] = 1.0
    is_done = np.all(combined_states, axis=-1, keepdims=True)
    absorbing_obs = np.where(is_done, absorbing_states, combined_states)
    return absorbing_obs


def fix_obs_normalization(input_obs: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """
    Linearly transforms the observations
    approximately mean equal zero and standard deviation equal one.
    """
    obs_mean = obs.mean(axis=0)
    obs_std = np.maximum(obs.std(axis=0), 0.001)
    return normalize(input_obs, obs_mean, obs_std)
