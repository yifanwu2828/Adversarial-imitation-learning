import re
import warnings
from pprint import pprint
from typing import Tuple, Dict, Union, Type, Sequence, Optional

import gym
import numpy as np
from gym.spaces import Box, Discrete, MultiDiscrete, MultiBinary

from ail.common.type_alias import GymEnv, GymWrapper, GymSpace, GymDict
from ail.color_console import COLORS
from ail.wrapper import EnvWrapper


def maybe_make_env(
    env: Union[GymEnv, str, None],
    env_wrapper: Optional[Sequence[Type[GymWrapper]]] = None,
    verbose=2,
    tag="",
    color="invisible",
) -> GymEnv:
    """
    If env is a string, make the environment; otherwise, return env.
    :param env: The environment to learn from.
    :param env_wrapper: Env wrapper
    :param tag:
    :param verbose:
    :param color:
    :return A Gym environment (maybe wrapped).
    """
    if env_wrapper is None:
        env_wrapper = []
    print("\n".join(["-" * 100]))
    if isinstance(env, str):
        if verbose > 0:
            print(COLORS[color] + f"| Creating {tag} env from the given name: '{env}'")
        env = gym.make(env)
    if env_wrapper:
        for wrap in env_wrapper:
            str_wrap = None
            if isinstance(wrap, str):
                try:
                    str_wrap = wrap
                    wrap = EnvWrapper[wrap]
                except KeyError:
                    raise KeyError(f"{wrap} is not a valid wrapper")
            env = wrap(env)
            if verbose > 0:
                msg = COLORS[color] + f"| Wrapping {tag} env with: "
                try:
                    print(msg + f"{wrap.class_name}")
                except AttributeError:
                    print(msg + f"{str_wrap}")
    if verbose == 2:
        env_summary(env, tag, verbose=True)

    return env


def env_summary(env: Union[GymEnv, str], tag="", verbose=False) -> Dict:
    """
    Obtain a summary of given env (space, dimensions, constrains)
    :param env: initialized env or env_id
    :param tag: usage of env (e.g. train or test)
    :param verbose: whether or not to print the summary
    :return: summary in Dict
    """
    if isinstance(env, str):
        env = gym.make(env)
    b = re.split(r"<", repr(env))
    wrapper = b[1:-2]
    env_id = re.split(r">", b[-1])[0]
    summary = {
        "env": repr(env),
        "env_id": env_id,
        "is_wrapped": len(wrapper) > 0,
        "wrapper": wrapper,
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "reward_range": env.reward_range,
    }
    try:
        summary.update({"max_ep_len": env._max_episode_steps})  # noqa
    except AttributeError:
        pass

    if verbose:
        print("-" * 100)
        title = f"{tag} env summary".title()
        print(f"{title : ^80}")
        pprint(summary, sort_dicts=False)
        print("-" * 100 + "\n")
    return summary


def unwrap_wrapper(
    env: GymEnv, wrapper_class: Type[gym.Wrapper]
) -> Optional[gym.Wrapper]:
    """
    Retrieve a ``VecEnvWrapper`` object by recursively searching.
    :param env: Environment to unwrap
    :param wrapper_class: Wrapper to look for
    :return: Environment unwrapped till ``wrapper_class`` if it has been wrapped with it
    """
    env_tmp = env
    while isinstance(env_tmp, gym.Wrapper):
        if isinstance(env_tmp, wrapper_class):
            return env_tmp
        env_tmp = env_tmp.env
    return None


def is_wrapped(env: Type[GymEnv], wrapper_class: Type[gym.Wrapper]) -> bool:
    """
    Check if a given environment has been wrapped with a given wrapper.
    :param env: Environment to check
    :param wrapper_class: Wrapper class to look for
    :return: True if environment has been wrapped with ``wrapper_class``.
    """
    return unwrap_wrapper(env, wrapper_class) is not None


def get_obs_shape(
    obs_space: GymSpace,
) -> Union[Tuple[int, ...], Dict[str, Tuple[int, ...]]]:
    """Get the shape of the observation (useful for the buffers)."""
    # Borrow from SB3
    if isinstance(obs_space, Box):
        return obs_space.shape
    elif isinstance(obs_space, Discrete):
        # Observation is an int
        return (1,)  # noqa
    elif isinstance(obs_space, MultiDiscrete):
        # Number of discrete features
        return (int(len(obs_space.nvec)),)  # noqa
    elif isinstance(obs_space, MultiBinary):
        # Number of binary features
        return (int(obs_space.n),)  # noqa
    elif isinstance(obs_space, (Dict, GymDict)):
        return {
            key: get_obs_shape(subspace) for (key, subspace) in obs_space.spaces.items()
        }
    else:
        raise NotImplementedError(f"{obs_space} observation space is not supported")


def get_act_dim(act_space: GymSpace) -> int:
    """Get the dimension of the action space."""
    # Borrow from SB3
    if isinstance(act_space, Box):
        return int(np.prod(act_space.shape))
    elif isinstance(act_space, Discrete):
        # Action is an int
        return 1
    elif isinstance(act_space, MultiDiscrete):
        # Number of discrete actions
        return int(len(act_space.nvec))
    elif isinstance(act_space, MultiBinary):
        # Number of binary actions
        return int(act_space.n)
    else:
        raise NotImplementedError(f"{act_space} action space is not supported")


def get_flat_obs_dim(obs_space: GymSpace) -> int:
    """
    Get the dimension of the observation space when flattened.
    It does not apply to image observation space.
    """
    # Borrow from SB3
    # See issue https://github.com/openai/gym/issues/1915
    # it may be a problem for Dict/Tuple spaces too...
    if isinstance(obs_space, MultiDiscrete):
        return sum(obs_space.nvec)
    else:
        # Use Gym internal method
        return gym.spaces.utils.flatdim(obs_space)


def get_env_dim(env) -> Tuple[int, int]:
    """
    Get the dimension of both the observation space and action space from given env.
    :param env:
    :return:
    """
    # Observation Dimension
    if isinstance(env.observation_space, GymDict):
        obs_dim = 0
        for k in env.observation_space:
            if k != "achieved_goal":
                obs_dim += env.observation_space[k].shape[0]
    else:
        # Are the observations images?
        is_img = is_image_space(env.observation_space, check_channels=False)
        obs_dim = (
            env.observation_space.shape
            if is_img
            else get_flat_obs_dim(env.observation_space)
        )
    # Action Dimension (Is this env continuous, or discrete?)
    act_dim = get_act_dim(env.action_space)
    return obs_dim, act_dim


def is_image_space(
    observation_space: GymSpace,
    check_channels: bool = False,
) -> bool:
    """
    Check if a observation space has the shape, limits and dtype
    of a valid image.
    The check is conservative, so that it returns False if there is a doubt.
    Valid images: RGB, RGBD, GrayScale with values in [0, 255]
    :param observation_space:
    :param check_channels: Whether to do or not the check for the number of channels.
        e.g., with frame-stacking, the observation space may have more channels than expected.
    :return:
    """
    # Borrow from SB3
    if isinstance(observation_space, Box) and len(observation_space.shape) == 3:
        # Check the type
        if observation_space.dtype != np.uint8:
            return False

        # Check the value range
        if np.any(observation_space.low != 0) or np.any(observation_space.high != 255):
            return False

        # Skip channels check
        if not check_channels:
            return True
        # Check the number of channels
        if is_image_space_channels_first(observation_space):
            n_channels = observation_space.shape[0]
        else:
            n_channels = observation_space.shape[-1]
        # RGB, RGBD, GrayScale
        return n_channels in [1, 3, 4]
    return False


def is_image_space_channels_first(observation_space: Box) -> bool:
    """
    Check if an image observation space
    is channels-first (CxHxW, True) or channels-last (HxWxC, False).
    Use a heuristic that channel dimension is the smallest of the three.
    If second dimension is smallest, raise an exception (no support).
    :param observation_space:
    :return: True if observation space is channels-first image, False if channels-last.
    """
    # Borrow from SB3
    smallest_dimension = np.argmin(observation_space.shape).item()
    if smallest_dimension == 1:
        warnings.warn(
            "Treating image space as channels-last, while second dimension was smallest of the three."
        )
    return smallest_dimension == 0
