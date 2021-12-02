import argparse
import sys
from typing import Optional

import gym
from gym.core import Env
import numpy as np
import matplotlib.pyplot as plt

from gym.spaces import Box, Dict

from stable_baselines3 import HerReplayBuffer, PPO, SAC
from sb3_contrib import TQC
from sb3_contrib.common.wrappers import TimeFeatureWrapper

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    StopTrainingOnRewardThreshold,
)


try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


ALGO = {
    "ppo": PPO,
    "sac": SAC,
    "tqc": TQC,
}

WRAPPER = {
    "time_feature": TimeFeatureWrapper,
    
}

custom_objects = {
            "learning_rate": 0.0,
            "lr_schedule": lambda _: 0.0,
            "clip_range": lambda _: 0.0,
}

def build_env(env_name, sparse_reward=False, wrapper_lst=()):
    env = gym.make(env_name)
    
    if hasattr(env, "reward_type"):
        env.reward_type = "sparse" if sparse_reward else "dense"
        print(f"Using {env.reward_type} reward")
    else:
        print("Cannot Change reward type.")
    
    for wrapper_name in wrapper_lst:
        env = WRAPPER[wrapper_name](env)
        print(f"Wrapping {env_name} with {wrapper_name}")
    
    return Monitor(env)
    

def sb3_eval_callback(eval_env, model_save_dir, log_dir = None, n_eval_episodes=20, eval_freq=10_000, reward_threshold=None):
    eval_callback_kwargs = dict(
        eval_env=eval_env,
        best_model_save_path = model_save_dir,         #f"./logs/{env_name}/{algo}/best_model",
        log_path=log_dir,                       #f"./logs/{env_name}/{algo}/results",
        n_eval_episodes=n_eval_episodes,
        eval_freq=eval_freq,
        verbose=1,
    )
    if reward_threshold is not None:
        assert isinstance(reward_threshold, (float, int))
        print(
            f"Stop training when the model reaches the reward threshold: {reward_threshold}"
        )
        # Stop training when the model reaches the reward threshold
        callback_on_best = StopTrainingOnRewardThreshold(
            reward_threshold=reward_threshold, verbose=1
        )
        eval_callback_kwargs["callback_on_new_best"] = callback_on_best
    
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(**eval_callback_kwargs)

    return eval_callback

def create_model(
    env: gym.Env,
    algo: str,
    device: str = "auto",
    learning_rate: float = 3e-4,
    learning_starts: int = 1_000,
):
    """
    Train the expert policy in RL
    """
    

    algo_cls = ALGO.get(algo.lower())
    
    if algo_cls is None:
        raise ValueError(f"RL algorithm {algo} not supported yet ...")

    
    print("Training from scratch ...\n")
        
    model = algo_cls(
        "MultiInputPolicy",
        env,
        learning_rate=learning_rate,
        learning_starts=learning_starts,
        replay_buffer_class=HerReplayBuffer,
        # Parameters for HER
        # replay_buffer_kwargs=dict(
        #     n_sampled_goal=4,
        #     goal_selection_strategy='future',
        #     online_sampling=True,
        #     max_episode_length=env.spec.max_episode_steps,
        # ),
        
        verbose=1,
    )

    return model


def save_policy(model, policy_name):
    model.save(policy_name)


def load_policy(algo, model_path, env, device="cpu"):
    algo_cls = ALGO.get(algo.lower())

    if algo_cls is None:
        raise ValueError(f"RL algorithm {algo} not supported yet ...")

    ic(model_path)
    model = algo_cls.load(model_path, env=env, device=device, custom_objects=custom_objects)
    ic(env.observation_space)
    return model


def visualize_policy(env, model, num_episodes=100, render=True, random_policy=False):
    """
    Visualize the policy in env
    """
    # Ensure testing on same device
    total_ep_returns = []
    total_ep_lengths = []
    total_success_rate = 0

    for _ in range(num_episodes):
        obs = env.reset()
        ep_ret, ep_len = 0.0, 0
        done = False

        while not done:
            if random_policy:
                action = env.action_space.sample()
            else:
                action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            reward = 0 if reward ==0 else reward
            ic(reward, done, info)
            ep_ret += reward
            ep_len += 1

            if render:
                try:
                    env.render()
                except KeyboardInterrupt:
                    sys.exit(0)
            if done:
                total_ep_returns.append(ep_ret)
                total_ep_lengths.append(ep_len)
                total_success_rate += info["is_success"]
                obs = env.reset()

    mean_episode_reward = np.mean(total_ep_returns)
    std_episode_reward = np.std(total_ep_lengths)
    print(f"-" * 50)
    print(
        f"Mean episode reward: {mean_episode_reward:.3f} +/- "
        f"{std_episode_reward:.3f} in {num_episodes} episodes "
        f"Success rate: {total_success_rate / num_episodes * 100:.2f}%"""
    )
    print(f"-" * 50)
    env.close()
    return total_ep_returns


def main():
    parser = argparse.ArgumentParser(description="Train the expert policy in RL")
    parser.add_argument(
        "--env_id",
        "-env",
        type=str,
        choices = ["FetchReach-v1", "FetchPush-v1", "FetchSlide-v1", "FetchPickAndPlace-v1"],
    )
    parser.add_argument("--algo", type=str, default="SAC")
    parser.add_argument("--num_steps", "-n", type=int, default=100_000)
    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-4)
    parser.add_argument("--learning_starts", "-start", type=float, default=1_000)
    parser.add_argument("--sparse", "-s", action="store_true")
    parser.add_argument("--reward_threshold", type=float)

    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--train", "-t", action="store_true")
    parser.add_argument("--render", "-r", action="store_true")
    parser.add_argument("--rnd",  action="store_true")
    
    args = parser.parse_args()

    env = build_env(args.env_id, sparse_reward=True, wrapper_lst=["time_feature"]) 
    eval_env = build_env(args.env_id, sparse_reward=True, wrapper_lst=["time_feature"])  
    
    ic(env.reward_type)
    ic(eval_env.reward_type)


    if args.train:
        model = create_model(
            env,
            args.algo,
            device= "auto",
            learning_rate= args.learning_rate,
            learning_starts= args.learning_starts,
        )
        
        callback = sb3_eval_callback(eval_env, model_save_dir=f"./logs/{args.env_id}/{args.algo.lower()}/best_model", eval_freq=1000, reward_threshold=args.reward_threshold)
        model.learn(total_timesteps=int(args.num_steps), log_interval=4, callback=callback)
       

    # Load policy from file

    model_path = f"./ail/rl-trained-agents/{args.env_id}/{args.algo.lower()}/{args.env_id}_sb3"
    model = load_policy(args.algo, model_path, env, device="cpu")

    # Evaluate the policy and Optionally render the policy
    total_ep_returns = visualize_policy(env, model, render=args.render, random_policy=args.rnd)
    plt.scatter(range(len(total_ep_returns)), total_ep_returns)
    plt.show()


if __name__ == "__main__":

    main()
    
    

        

