import argparse
import pathlib

import gym
import numpy as np
import torch as th
import matplotlib.pyplot as plt
from tqdm import tqdm
from stable_baselines3 import SAC, PPO

from ail.agents import ALGO
from ail.common.utils import set_random_seed
from ail.common.env_utils import maybe_make_env

try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa


SB3_ALGO = {
    "sb3_ppo": PPO,
    "sb3_sac": SAC,
}


def eval_th_algo(model, env_id, num_episodes=10, seed=42, render=False, sb3=False):
    """
    Evaluate a RL agent
    :param model: (BaseRLModel object) the RL Agent
    :param num_episodes: (int) number of episodes to evaluate it
    :param env_id: env name
    :param render: whether to render
    """
    # This function will only work for a single Environment
    eval_env = gym.make(env_id)
    eval_env.seed(seed)
    all_episode_rewards = []
    for _ in tqdm(range(num_episodes)):
        episode_rewards = []
        done = False
        obs = eval_env.reset()
        while not done:
            if sb3:
                action, _ = model.predict(th.as_tensor(obs).float())
            else:
                action = model.exploit(th.as_tensor(obs).float())
            # here, action, rewards and dones are arrays
            obs, reward, done, info = eval_env.step(np.asarray(action))
            episode_rewards.append(reward)
            if render:
                try:
                    eval_env.render()
                except KeyboardInterrupt:
                    pass

        all_episode_rewards.append(sum(episode_rewards))
    eval_env.close()

    mean_episode_reward = np.mean(all_episode_rewards)
    std_episode_reward = np.std(all_episode_rewards)
    print(f"-" * 50)
    print(
        f"Mean episode reward: {mean_episode_reward:.3f} +/- "
        f"{std_episode_reward:.3f} in {num_episodes} episodes"
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
        choices=["InvertedPendulum-v2", "HalfCheetah-v2", "Hopper-v3"],
        help="Envriment to test on",
    )

    p.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["ppo", "sac", "sb3_ppo", "sb3_sac"],
        help="RL algo to test",
    )
    # Policy Arch
    p.add_argument("--n_layers", "-l", type=int, default=2)
    p.add_argument("--size", "-s", type=int, default=64)

    p.add_argument("--rollout_length", type=int, default=None)
    p.add_argument("--num_eval_episodes", type=int, default=20)
    p.add_argument("--render", "-r", action="store_true")
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.seterr(all="raise")
    th.autograd.set_detect_anomaly(True)

    # Set random seed
    set_random_seed(args.seed)

    # dummy arguments
    dummy_env = maybe_make_env(args.env_id)

    # run only on cpu for testing
    device = th.device("cpu")

    # Path
    path = pathlib.Path.cwd()
    print(f"current_dir: {path}")

    if not args.weight:
        demo_dir = path.parent / "rl-trained-agents" / args.env_id
        if args.algo.startswith("sb3"):
            args.weight = demo_dir / args.algo[4:] / f"{args.env_id}_sb3"
        else:
            args.weight = demo_dir / args.algo / f"{args.env_id}_actor.pth"

    if args.algo.startswith("sb3"):
        demo = SB3_ALGO[args.algo].load(args.weight)
        use_sb3 = True
    else:
        pi_arch = [args.size] * args.n_layers
        demo = ALGO[args.algo].load(
            path=args.weight,
            device=device,
            policy_kwargs={"pi": pi_arch},  # * make sure the arch matches state_dict
            env=dummy_env,
        )
        demo.actor.eval()
        use_sb3 = False

    print(f"weight_dir: {args.weight}\n")

    # Max episode length
    max_ep_len = (
        args.rollout_length if args.rollout_length else dummy_env._max_episode_steps
    )
    total_steps = args.num_eval_episodes * max_ep_len

    eval_th_algo(
        model=demo,
        env_id=args.env_id,
        num_episodes=args.num_eval_episodes,
        seed=args.seed,
        render=args.render,
        sb3=use_sb3,
    )
