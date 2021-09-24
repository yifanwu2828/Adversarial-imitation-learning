import argparse
from datetime import datetime
import os
import time
import gym
import yaml

import numpy as np
import torch as th
from icecream import ic

from stable_baselines3 import A2C, SAC, PPO, HER
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

from ail.common.env_utils import maybe_make_env


ALGO = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
    "her": HER,
}


def train_demo(params: dict, algo: str, eval_render=False):
    print(f"{'-'*15} Collecting SB3_{algo.upper()} Demo {'-'*15} ")
    ic(params)

    model_cls = ALGO[algo]

    # Separate evaluation env
    eval_env = Monitor(gym.make(params["env_id"]))
    # Use deterministic actions for evaluation
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=params["save_dir"],
        log_path=params["save_dir"],
        eval_freq=params["eval_freq"],
        deterministic=True,
        render=eval_render,
    )

    if algo == "ppo":
        # Parallel environments
        vec_env = make_vec_env(params["env_id"], n_envs=params["n_envs"])
        sb3_model = model_cls(
            # key
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=params.get("learning_rate", 3e-4),
            n_steps=params.get("n_steps", 2048),
            batch_size=params.get("batch_size", 64),  # 64
            n_epochs=params.get("n_epochs", 10),
            gamma=params.get("gamma", 0.99),
            gae_lambda=params.get("gae_lambda", 0.95),
            clip_range=params.get("clip_range", 0.2),
            clip_range_vf=params.get("clip_range_vf", None),
            ent_coef=params.get("ent_coef", 0.0),
            vf_coef=params.get("vf_coef", 0.5),
            max_grad_norm=params.get("max_grad_norm", 0.5),
            use_sde=params.get("use_sde", False),
            sde_sample_freq=params.get("sde_sample_freq", -1),
            target_kl=None,
            policy_kwargs=params.get("policy_kwargs", None),
            # utils
            tensorboard_log=params.get("tensorboard_log", None),
            create_eval_env=params.get("create_eval_env", False),
            verbose=params.get("verbose", 1),
            seed=params.get("seed", 42),
            device=params.get("device", "auto"),
        )
    elif algo == "sac":
        train_env = maybe_make_env(params["env_id"], env_wrapper=None, verbose=2)
        sb3_model = model_cls(
            # key
            policy="MlpPolicy",
            env=train_env,
            learning_rate=params.get("learning_rate", 3e-4),
            buffer_size=params.get("buffer_size", int(3e6)),
            learning_starts=params.get("learning_starts", 10_000),
            batch_size=params.get("batch_size", 256),
            # the soft update coefficient ("Polyak update", between 0 and 1)
            tau=params.get("tau", 0.005),
            gamma=params.get("gae_lambda", 0.99),
            # Update the model every ``train_freq`` steps.
            train_freq=params.get("train_freq", 1),
            # How many gradient steps to do after each rollout
            gradient_steps=params.get("gradient_steps", 1),
            action_noise=params.get("action_noise", None),
            optimize_memory_usage=params.get("optimize_memory_usage", False),
            ent_coef="auto",
            # update the target network every ``target_network_update_freq``gradient steps.
            target_update_interval=params.get("target_update_interval", 2),
            target_entropy="auto",
            use_sde=False,
            sde_sample_freq=-1,
            use_sde_at_warmup=False,
            # Whether to use gSDE instead of uniform sampling during the warm up phase (before learning starts)
            # utils
            tensorboard_log=params.get("tensorboard_log", None),
            create_eval_env=params.get("create_eval_env", False),
            policy_kwargs=params.get("policy_kwargs", None),
            verbose=params.get(
                "verbose", 1
            ),  # verbosity level: 0 no output, 1 info, 2 debug
            seed=params.get("seed", 42),
            device=params.get("device", "auto"),
        )
    else:
        raise RuntimeError("ALGO Not Found")

    start_time = time.time()
    sb3_model.learn(
        total_timesteps=params["n_timesteps"]
        + 1,  # save last one if with proper eval interval
        log_interval=params["log_interval"],
        callback=eval_callback,
    )
    print(f"Finish in {time.time() - start_time}")

    # Save model
    print(f"Saving model to {params['final_model_dir']}")
    sb3_model.save(params["final_model_dir"])


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env_id", "-env", type=str, default="Pendulum-v0"
    )  # HalfCheetah-v2
    p.add_argument("--env_wrapper", "-w", type=str, default="")
    p.add_argument("--algo", "-algo", type=str, default="sac")
    p.add_argument("--render", "-r", action="store_true", default=False)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_dir", "-log", type=str, default="./runs")
    p.add_argument("--device", type=str, default="auto")
    args = p.parse_args()

    time_now = datetime.now().strftime("%Y%m%d-%H%M")

    tb_log = os.path.join(args.log_dir, f"{args.env_id}/summary/{args.algo}/")
    save_dir = os.path.join(args.log_dir, f"{args.env_id}/model/{args.algo}/{time_now}")
    for path in [tb_log, save_dir]:
        if not os.path.exists(path):
            os.makedirs(path)

    params = {
        "algo": args.algo,
        "env_id": args.env_id,
        "tensorboard_log": tb_log,
        "save_dir": save_dir,
        "final_model_dir": f"../weights/{args.env_id}/sb3_{args.algo}",
        "device": args.device,
        "eval_freq": 10_000,
    }

    # path to save final model
    if not os.path.exists(f"../weights/{args.env_id}"):
        os.makedirs(f"../weights/{args.env_id}")

    # PPO
    ppo_hyperparams = dict(
        normalize=True,
        n_envs=1,
        n_timesteps=20_000,
        policy="MlpPolicy",
        batch_size=256,
        n_steps=8,
        gamma=0.99,
        learning_rate=7.77e-05,
        ent_coef=0.00429,
        clip_range=0.1,
        n_epochs=10,
        gae_lambda=0.95,
        max_grad_norm=5,
        vf_coef=0.19,
        use_sde=False,
        policy_kwargs=dict(log_std_init=-3.29, ortho_init=False),
    )

    # SAC
    sac_hyperparams = dict(
        n_timesteps=2e6,
        lr=7e-4,
        batch_size=256,
        buffer_size=int(300_000),
        gamma=0.99,
        learning_starts=10_000,  # how many steps of the model to collect transitions for before learning starts
        tau=0.02,  # soft update coefficient ("Polyak update", between 0 and 1)
        train_freq=8,
        gradient_steps=8,  # How many gradient steps to do after each rollout
        target_update_interval=1,
        action_noise=None,
        ent_coef="auto",
        target_entropy="auto",
        use_sde=False,
        sde_sample_freq=-1,
        use_sde_at_warmup=False,
        verbose=2,
        log_interval=4,
        policy_kwargs=dict(log_std_init=-3, net_arch=[128, 128]),  # best 128:128
    )

    if args.algo.lower() == "ppo":
        hyperparams = ppo_hyperparams
    elif args.algo.lower() == "sac":
        hyperparams = sac_hyperparams
    else:
        raise RuntimeError("ALGO Not Found")
    params.update(hyperparams)

    train_demo(params, algo=args.algo, eval_render=args.render)
    with open(os.path.join(save_dir, "hyperparams.yaml"), "w") as f:
        yaml.dump(params, f)
