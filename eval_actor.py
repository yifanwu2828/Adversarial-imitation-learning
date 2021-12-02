import argparse
import pathlib

import numpy as np
import torch as th
import matplotlib.pyplot as plt
from tqdm import tqdm

from ail.agents import ALGO
from ail.common.utils import set_random_seed
from ail.common.env_utils import maybe_make_env

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa



def eval_th_algo(model, eval_env, num_episodes=10, seed=42, render=False):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :param env_id: env name
    :param render: whether to render
    """
    # This function will only work for a single Environment
    
    eval_env.seed(seed)
    all_episode_rewards = []
    total_success_rate = 0
    for _ in tqdm(range(num_episodes)):
        episode_rewards = []
        done = False
        obs = eval_env.reset()
        while not done:
            action = model.exploit(th.as_tensor(obs).float(), scale=1)
            # here, action, rewards and dones are arrays
            obs, reward, done, info = eval_env.step(np.asarray(action))
            episode_rewards.append(reward)
            if render:
                try:
                    eval_env.render()
                except KeyboardInterrupt:
                    pass

        all_episode_rewards.append(sum(episode_rewards))
        total_success_rate += info["is_success"]
    eval_env.close()

    mean_episode_reward = np.mean(all_episode_rewards)
    std_episode_reward = np.std(all_episode_rewards)
    print(f"-" * 50)
    print(
        f"Mean episode reward: {mean_episode_reward:.3f} +/- "
        f"{std_episode_reward:.3f} in {num_episodes} episodes "
        f"Success rate: {total_success_rate / num_episodes * 100:.2f}%"""
    )
    plt.plot(all_episode_rewards)
    plt.show()
    return all_episode_rewards


if __name__ == "__main__":

    p = argparse.ArgumentParser()
    p.add_argument("--weight", type=str, default="")
    p.add_argument(
        "--env_id",
        type=str,
        required=True,
        choices=["FetchReach-v1", "FetchPush-v1", "FetchSlide-v1", "FetchPickAndPlace-v1"],
        help="Envriment to test on",
    )

    p.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["ppo", "sac"],
        help="RL algo to test",
    )
    # Policy Arch
    p.add_argument("--n_layers", "-l", type=int, default=2)
    p.add_argument("--size", "-s", type=int, default=64)

    p.add_argument("--rollout_length", type=int, default=None)
    p.add_argument("--num_eval_episodes", type=int, default=50)
    p.add_argument("--render", "-r", action="store_true")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.seterr(all="raise")
    th.autograd.set_detect_anomaly(True)

    # Set random seed
    set_random_seed(args.seed)

    env_wrapper = ["clip_act", "filter_obs", "flatten_obs", "time_aware", "done_success"]

    # dummy arguments
    dummy_env = maybe_make_env(args.env_id, env_wrapper=env_wrapper)
    ic(dummy_env)
    # run only on cpu for testing
    device = th.device("cpu")

    # Path
    path = pathlib.Path.cwd()
    print(f"current_dir: {path}")


    args.weight = "./gen_actor.pth"
    
    pi_arch = [args.size] * args.n_layers
    demo = ALGO[args.algo].load(
        path=args.weight,
        device=device,
        policy_kwargs={"pi": pi_arch},  # * make sure the arch matches state_dict
        env=dummy_env,
    )
    demo.actor.eval()

    print(f"weight_dir: {args.weight}\n")

    eval_th_algo(
        model=demo,
        eval_env=dummy_env,
        num_episodes=args.num_eval_episodes,
        seed=args.seed,
        render=args.render,
    )
