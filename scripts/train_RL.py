import os
import pathlib
import sys
import argparse

import yaml
import torch as th

from ail.trainer import Trainer
from ail.common.utils import make_unique_timestamp
from config.config import get_cfg_defaults


try:
    from icecream import install  # noqa

    install()
except ImportError:  # Graceful fallback if IceCream isn't installed.
    ic = lambda *a: None if not a else (a[0] if len(a) == 1 else a)  # noqa

try:
    from dotenv import load_dotenv, find_dotenv  # noqa

    load_dotenv(find_dotenv())

except ImportError:
    pass


def CLI():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--env_id",
        type=str,
        choices=["InvertedPendulum-v2", "HalfCheetah-v2", "Hopper-v3", "NavEnv-v0"],
        help="Envriment to train on",
    )
    p.add_argument(
        "--algo",
        type=str,
        default="sac",
        choices=[
            "ppo",
            "sac",
        ],
        help="RL algo to use",
    )
    p.add_argument("--num_steps", "-n", type=int, default=0.5 * 1e6)
    p.add_argument("--rollout_length", type=int, default=None)
    p.add_argument("--batch_size", "-bs", type=int, default=256)
    # p.add_argument("--buffer_size", type=int, default=1 * 1e6)
    p.add_argument("--log_every_n_updates", "-lg", type=int, default=20)
    p.add_argument("--eval_interval", type=int, default=5 * 1e3)
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument("--save_freq", type=int, default=50_000)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--verbose", type=int, default=2)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--use_wandb", "-wb", action="store_true")

    args = p.parse_args()

    if args.env_id == "NavEnv-v0":
        import gym_nav  # noqa

    args.device = "cuda" if args.cuda else "cpu"

    # Enforce type int
    args.num_steps = int(args.num_steps)
    args.batch_size = int(args.batch_size)
    args.log_every_n_updates = int(args.log_every_n_updates)

    # How often (in terms of steps) to output training info.
    args.log_interval = args.batch_size * args.log_every_n_updates

    return args


def run(args, cfg, path):
    """Training Configuration"""
    algo_kwargs = dict(
        # common args
        device=args.device,
        fp16=args.fp16,
        seed=args.seed,
        gamma=cfg.ALGO.gamma,
        max_grad_norm=cfg.ALGO.max_grad_norm,
        optim_kwargs=dict(cfg.OPTIM),
    )

    rl_algo = args.algo.lower()

    if rl_algo == "ppo":
        # state_ space, action space inside trainer
        ppo_kwargs = dict(
            # buffer args
            batch_size=args.batch_size,  # PPO assums batch size == buffer_size
            buffer_kwargs=dict(
                with_reward=cfg.PPO.with_reward, extra_data=cfg.PPO.extra_data
            ),
            # PPO only args
            epoch_ppo=cfg.PPO.epoch_ppo,
            gae_lambda=cfg.PPO.gae_lambda,
            clip_eps=cfg.PPO.clip_eps,
            coef_ent=cfg.PPO.coef_ent,
            # poliy args: net arch, activation, lr
            policy_kwargs=dict(
                pi=cfg.PPO.pi,
                vf=cfg.PPO.vf,
                activation=cfg.PPO.activation,
                critic_type=cfg.PPO.critic_type,
                lr_actor=cfg.PPO.lr_actor,
                lr_critic=cfg.PPO.lr_critic,
                orthogonal_init=cfg.PPO.orthogonal_init,
            ),
        )
        algo_kwargs.update(ppo_kwargs)
        sac_kwargs = None

    elif rl_algo == "sac":
        sac_kwargs = dict(
            # buffer args
            batch_size=args.batch_size,
            buffer_size=cfg.SAC.buffer_size,
            buffer_kwargs=dict(
                with_reward=cfg.SAC.with_reward, extra_data=cfg.SAC.extra_data
            ),
            # SAC only args
            start_steps=cfg.SAC.start_steps,
            lr_alpha=cfg.SAC.lr_alpha,
            log_alpha_init=cfg.SAC.log_alpha_init,
            tau=cfg.SAC.tau,
            # * Recommend to sync following two params to reduce overhead
            num_gradient_steps=cfg.SAC.num_gradient_steps,  # ! slow O(n)
            target_update_interval=cfg.SAC.target_update_interval,
            # poliy args: net arch, activation, lr
            policy_kwargs=dict(
                pi=cfg.SAC.pi,
                qf=cfg.SAC.qf,
                activation=cfg.SAC.activation,
                critic_type=cfg.SAC.critic_type,
                lr_actor=cfg.SAC.lr_actor,
                lr_critic=cfg.SAC.lr_critic,
            ),
        )
        if "absorbing" in cfg.ENV.wrapper:
            sac_kwargs["use_absorbing_state"] = True
        algo_kwargs.update(sac_kwargs)
        ppo_kwargs = None

    else:
        raise ValueError(f"RL ALgo {args.algo} not Implemented.")

    timestamp = make_unique_timestamp()
    exp_name = os.path.join(args.env_id, args.algo, f"seed{args.seed}-{timestamp}")
    log_dir = path.joinpath("runs", exp_name)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    config = dict(
        total_timesteps=args.num_steps,
        env=args.env_id,
        algo=args.algo,
        algo_kwargs=algo_kwargs,
        env_kwargs={"env_wrapper": cfg.ENV.wrapper},
        test_env_kwargs={"env_wrapper": cfg.TEST_ENV.wrapper},
        max_ep_len=args.rollout_length,
        seed=args.seed,
        eval_interval=args.eval_interval,
        num_eval_episodes=args.num_eval_episodes,
        save_freq=args.save_freq,
        log_dir=log_dir,
        log_interval=args.log_interval,
        verbose=args.verbose,
        use_wandb=args.use_wandb,
        wandb_kwargs=cfg.WANDB,
    )

    # Log with tensorboard and sync to wandb dashboard as well
    # https://docs.wandb.ai/guides/integrations/tensorboard
    if args.use_wandb:
        try:
            import wandb

            tags = ["baseline", f"{args.env_id}", str(args.algo).upper()]
            if "absorbing" in cfg.ENV.wrapper:
                tags.append("absorbing")
            # Save API key for convenience or you have to login every time
            wandb.login()
            wandb.init(
                project="AIL",
                notes="tweak baseline",
                tags=tags,
                config=config,  # Hyparams & meta data
            )
            wandb.run.name = exp_name
            # make sure to use the same config as passed to wandb
            config = wandb.config
        except ImportError:
            print("`wandb` Module Not Found")
            sys.exit(0)

    # Create Trainer
    trainer = Trainer(**config)

    # algo kwargs
    print("-" * 35, f"{args.algo}", "-" * 35)
    ic(algo_kwargs)

    # Saving hyperparams to yaml file
    with open(os.path.join(log_dir, "hyperparams.yaml"), "w") as f:
        yaml.dump(algo_kwargs, f)

    del algo_kwargs, ppo_kwargs, sac_kwargs

    trainer.run_training_loop()


if __name__ == "__main__":
    # ENVIRONMENT VARIABLE
    os.environ["WANDB_NOTEBOOK_NAME"] = "test"  # modify to assign a meaningful name

    args = CLI()

    path = pathlib.Path(__file__).parent.resolve()
    print(path)

    cfg_path = path / "config" / "rl_configs.yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    print(cfg)

    if args.debug:
        import numpy as np

        np.seterr(all="raise")
        th.autograd.set_detect_anomaly(True)

    if args.cuda:
        # TODO: investigate this
        os.environ["OMP_NUM_THREADS"] = "1"
        # torch backends
        th.backends.cudnn.benchmark = (
            cfg.CUDA.cudnn
        )  # ? Does this useful for non-convolutions?

    run(args, cfg, path)
