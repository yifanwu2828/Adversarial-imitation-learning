from typing import Tuple, List, Union

import gym
import numpy as np
import torch as th
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm


def to_torch(
    array: Union[np.ndarray, Tuple, List],
    device: Union[th.device, str],
    copy: bool = True,
) -> th.Tensor:
    """
    Convert a numpy array to a PyTorch tensor.
    Note: it copies the data by default
    :param array:
    :param device: PyTorch device to which the values will be converted
    :param copy: Whether to copy or not the data
        (may be useful to avoid changing things be reference)
    :return:
    """
    if copy:
        return th.tensor(array, dtype=th.float32).to(device)
    elif isinstance(array, np.ndarray):
        return from_numpy(array, device)
    else:
        return th.as_tensor(array, dtype=th.float32).to(device)


def from_numpy(array: np.ndarray, device) -> th.Tensor:
    """Convert numpy array to torch tensor  and send to device('cuda:0' or 'cpu')"""
    return th.from_numpy(array).float().to(device)


def to_numpy(tensor: th.Tensor, flatten=False) -> np.ndarray:
    """Convert torch tensor to numpy array and send to CPU"""
    if flatten:
        tensor.squeeze_(-1)
    return tensor.detach().cpu().numpy()


def get_statistics(x: np.ndarray):
    return x.mean(), x.std(), x.min(), x.max()


def get_mean_std(rew, name="true", verbose=False):
    """Calculate Mean and Standard Deviation"""
    if not isinstance(rew, np.ndarray):
        rew = np.array(rew)
    mean_rew = rew.mean()
    std_rew = rew.std()
    if verbose:
        print(f"mean_{name}_reward:{mean_rew:.3f}, std_{name}_reward:{std_rew:.3f}")
    return mean_rew, std_rew


def get_metrics(rew1, rew2, verbose=False):
    """Metrics: MAE, MSE, RMS, R2"""
    mae = mean_absolute_error(rew1, rew2)
    mse = mean_squared_error(rew1, rew2)
    rms = mean_squared_error(rew1, rew2, squared=False)
    r2 = r2_score(rew1, rew2)
    if verbose:
        print(f"MAE: {mae :.3f}")
        print(f"MSE: {mse :.3f}")
        print(f"RMS: {rms :.3f}")
        print(f"R2: {r2 :.3f}")
    return mae, mse, rms, r2


def evaluate(model, env_id, num_episodes=100, render=False):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :param env_id: env name
    :param render: whether to render
    """
    # This function will only work for a single Environment
    eval_env = gym.make(env_id)
    all_episode_rewards = []
    all_episode_length = []
    for _ in tqdm(range(num_episodes)):
        episode_rewards = []
        episode_length = 0
        done = False
        obs = eval_env.reset()
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            # here, action, rewards and dones are arrays
            obs, reward, done, info = eval_env.step(action)
            episode_rewards.append(reward)
            episode_length += 1
            if render:
                try:
                    eval_env.render()
                except KeyboardInterrupt:
                    pass

        all_episode_length.append(episode_length)
        all_episode_rewards.append(sum(episode_rewards))
    eval_env.close()

    mean_episode_length = np.mean(all_episode_length)
    mean_episode_reward = np.mean(all_episode_rewards)
    std_episode_reward = np.std(all_episode_rewards)
    max_episode_reward = np.max(all_episode_rewards)
    min_episode_reward = np.min(all_episode_rewards)
    print(f"-" * 50)
    print(f"Mean episode length: {mean_episode_length }")
    print(
        f"Mean episode reward: {mean_episode_reward:.3f} +/- "
        f"{std_episode_reward:.3f} in {num_episodes} episodes"
    )
    print(f"Max episode reward: {max_episode_reward:.3f}")
    print(f"Min episode reward: {min_episode_reward:.3f}")
    return (
        all_episode_rewards,
        mean_episode_reward,
        std_episode_reward,
        max_episode_reward,
        min_episode_reward,
    )
