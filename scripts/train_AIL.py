import os
import pathlib
import sys
import argparse
from copy import deepcopy

import yaml
import numpy as np
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
        "-env",
        type=str,
        choices=["InvertedPendulum-v2", "HalfCheetah-v2", "Hopper-v3", "NavEnv-v0"],
        help="Envriment to train on.",
    )
    p.add_argument(
        "--algo",
        type=str,
        choices=[
            "airl",
            "gail",
        ],
        required=True,
        help="Adversarial imitation algo to use.",
    )
    p.add_argument(
        "--gen_algo",
        type=str,
        choices=[
            "ppo",
            "sac",
        ],
        required=True,
        help="RL algo to use as generator.",
    )
    p.add_argument(
        "--demo_path", "-demo", type=str, help="Path to demo"
    )  # required=True,

    # Total steps and batch size
    p.add_argument("--num_steps", "-n", type=int, default=2 * 1e6)
    p.add_argument("--rollout_length", "-ep_len", type=int, default=None)
    p.add_argument("--gen_batch_size", "-gbs", type=int, default=1_024)
    p.add_argument("--replay_batch_size", "-rbs", type=int, default=1_024)

    # Logging and evaluation
    p.add_argument("--log_every_n_updates", "-lg", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=5 * 1e3)
    p.add_argument("--eval_mode", type=str, default="average")
    p.add_argument("--num_eval_episodes", type=int, default=10)
    p.add_argument(
        "--save_freq",
        type=int,
        default=50_000,
        help="Save model every `save_freq` steps.",
    )

    # Cuda options
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--fp16", action="store_true")

    # Utility
    p.add_argument("--verbose", type=int, default=2)
    p.add_argument("--debug", action="store_true")
    p.add_argument("--use_wandb", "-wb", action="store_true")
    args = p.parse_args()

    if args.env_id == "NavEnv-v0":
        import gym_nav  # noqa

    args.device = "cuda" if args.cuda else "cpu"

    # Enforce type int
    args.num_steps = int(args.num_steps)
    args.gen_batch_size = int(args.gen_batch_size)
    args.log_every_n_updates = int(args.log_every_n_updates)

    # How often (in terms of steps) to output training info.
    args.log_interval = args.gen_batch_size * args.log_every_n_updates
    return args


def run(args, cfg, path):
    """Training Configuration."""

    algo_kwargs = dict(
        # common args
        device=args.device,
        fp16=args.fp16,
        seed=cfg.ALGO.seed,
        gamma=cfg.ALGO.gamma,
        max_grad_norm=cfg.ALGO.max_grad_norm,
        optim_kwargs=dict(cfg.OPTIM),
    )

    gen_algo = args.gen_algo.lower()
    if gen_algo == "ppo":
        # state space, action space inside trainer
        ppo_kwargs = dict(
            # buffer args
            batch_size=args.gen_batch_size,  # PPO assums batch size == buffer_size
            buffer_kwargs=dict(with_reward=False, extra_data=["log_pis"]),
            # PPO only args
            epoch_ppo=cfg.PPO.epoch_ppo,
            gae_lambda=cfg.PPO.gae_lambda,
            clip_eps=cfg.PPO.clip_eps,
            coef_ent=cfg.PPO.coef_ent,
            # poliy args: net arch, activation, lr.
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
        gen_kwargs = {**algo_kwargs, **ppo_kwargs}
        sac_kwargs = None

    elif gen_algo == "sac":
        sac_kwargs = dict(
            # buffer args.
            batch_size=args.gen_batch_size,  # PPO assums batch size == buffer_size
            buffer_size=cfg.SAC.buffer_size,  # only used in SAC,
            buffer_kwargs=dict(with_reward=False, extra_data=cfg.SAC.extra_data),
            # SAC only args.
            start_steps=cfg.SAC.start_steps,
            lr_alpha=cfg.SAC.lr_alpha,
            log_alpha_init=cfg.SAC.log_alpha_init,
            tau=cfg.SAC.tau,  # 0.005
            # * Recommend to sync following two params to reduce overhead.
            num_gradient_steps=cfg.SAC.num_gradient_steps,  # ! slow O(n)
            target_update_interval=cfg.SAC.target_update_interval,
            use_as_generator=True,
            # poliy args: net arch, activation, lr.
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
        gen_kwargs = {**algo_kwargs, **sac_kwargs}
        ppo_kwargs = None

    else:
        raise ValueError(f"RL ALgo (generator) {args.gen_algo} not Implemented.")

    # Demo data.
    if args.demo_path is None:
        # TODO: REMOVE THIS
        demo_dir = path / "transitions" / args.env_id

        if "absorbing" in cfg.ENV.wrapper:
            demo_dir = demo_dir / "wrapped" / "AbsorbingWrapper_"
        dir_lst = os.listdir(demo_dir)
        repeat = []
        for fname in dir_lst:
            if fname.endswith("npz"):
                repeat.append(fname)
        if len(repeat) > 1:
            raise ValueError(f"Too many demo files in {demo_dir}")
        args.demo_path = demo_dir / fname

    transitions = dict(np.load(args.demo_path))

    algo_kwargs.update(
        dict(
            replay_batch_size=args.replay_batch_size,
            buffer_exp="replay",
            buffer_kwargs=dict(
                with_reward=False,
                transitions=transitions,  # * transitions must be a dict-like object
            ),
            gen_algo=args.gen_algo,
            gen_kwargs=gen_kwargs,
            disc_cls=cfg.AIRL.disc_cls,
            disc_kwargs=dict(
                hidden_units=cfg.DISC.hidden_units,
                hidden_activation=cfg.DISC.hidden_activation,
                gamma=cfg.ALGO.gamma,
                use_spectral_norm=cfg.DISC.use_spectral_norm,
                dropout_input=cfg.DISC.dropout_input,
                dropout_input_rate=cfg.DISC.dropout_input_rate,
                dropout_hidden=cfg.DISC.dropout_hidden,
                dropout_hidden_rate=cfg.DISC.dropout_hidden_rate,
                inverse=cfg.GAIL.inverse,
            ),
            disc_ent_coef=cfg.DISC.ent_coef,
            epoch_disc=cfg.DISC.epoch_disc,
            lr_disc=cfg.DISC.lr_disc,
            subtract_logp=cfg.AIRL.subtract_logp,
            rew_input_choice=cfg.DISC.rew_input_choice,
            rew_clip=cfg.DISC.rew_clip,
            max_rew_magnitude=cfg.DISC.max_rew_magnitude,
            min_rew_magnitude=cfg.DISC.min_rew_magnitude,
            use_absorbing_state="absorbing" in cfg.ENV.wrapper,
            infinite_horizon=cfg.DISC.infinite_horizon,
        )
    )

    timestamp = make_unique_timestamp()
    exp_name = os.path.join(args.env_id, args.algo, f"seed{cfg.ALGO.seed}-{timestamp}")
    log_dir = path.joinpath("runs", exp_name)

    config = dict(
        total_timesteps=args.num_steps,
        env=args.env_id,
        algo=args.algo,
        algo_kwargs=algo_kwargs,
        env_kwargs={"env_wrapper": cfg.ENV.wrapper},
        test_env_kwargs={"env_wrapper": cfg.TEST_ENV.wrapper},
        max_ep_len=args.rollout_length,
        seed=cfg.ALGO.seed,
        eval_interval=args.eval_interval,
        eval_behavior_type=args.eval_mode,
        num_eval_episodes=args.num_eval_episodes,
        save_freq=args.save_freq,
        log_dir=log_dir,
        log_interval=args.log_interval,
        verbose=args.verbose,
        use_wandb=args.use_wandb,
        wandb_kwargs=cfg.WANDB,
    )

    # Log with tensorboard and sync to wandb dashboard as well.
    # https://docs.wandb.ai/guides/integrations/tensorboard
    if args.use_wandb:
        # Not to store expert data in wandb.
        config_copy = deepcopy(config)
        config_copy["algo_kwargs"]["buffer_kwargs"].pop("transitions")
        # Remove unnecessary fields.
        entries_to_remove = (
            "num_eval_episodes",
            "eval_interval",
            "eval_behavior_type",
            "log_dir",
            "log_interval",
            "save_freq",
            "verbose",
            "use_wandb",
            "wandb_kwargs",
        )
        for k in entries_to_remove:
            config_copy.pop(k)

        # import pandas as pd
        # df = pd.json_normalize(config_copy, sep='_').to_dict(orient='records')[0]
        # config_copy={k.replace("algo_kwargs_", ""): v for k, v in df.items()}

        try:
            import wandb

            tags = [
                f"{args.env_id}",
                str(args.algo).upper(),
                str(args.gen_algo).upper(),
                str(cfg.DISC.rew_input_choice),
                str(cfg.OPTIM.optim_cls),
            ]
            if "absorbing" in cfg.ENV.wrapper:
                tags.append("absorbing")
            # Save API key for convenience or you have to login every time.
            wandb.login()
            wandb.init(
                entity="ucsd-erl-ail",
                project="ail",
                notes="tweak baseline",
                tags=tags,
                config=config_copy,  # Hyparams & meta data.
            )
            wandb.run.name = exp_name
        except ImportError:
            print("`wandb` Module Not Found")
            sys.exit(0)

    # Create Trainer.
    trainer = Trainer(**config)

    # It's a dict of data too large to store.
    algo_kwargs["buffer_kwargs"].pop("transitions")
    if args.verbose:
        # algo kwargs
        print("-" * 35, f"{args.algo}", "-" * 35)
        ic(algo_kwargs)

    # Saving hyperparams to yaml file.
    with open(os.path.join(log_dir, "hyperparams.yaml"), "w") as f:
        yaml.dump(algo_kwargs, f)

    del algo_kwargs, gen_kwargs, ppo_kwargs, sac_kwargs

    trainer.run_training_loop()


def main():
    args = CLI()

    # Path
    path = pathlib.Path(__file__).parent.resolve()
    print(f"File_dir: {path}")

    cfg_path = path / "config" / "ail_configs.yaml"
    cfg = get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    print(cfg)

    if args.debug:
        np.seterr(all="raise")
        th.autograd.set_detect_anomaly(True)

    if args.cuda:
        os.environ["OMP_NUM_THREADS"] = "1"
        # torch backends
        th.backends.cudnn.benchmark = (
            cfg.CUDA.cudnn
        )  # ? Does this useful for non-convolutions?
    else:
        os.environ["OMP_NUM_THREADS"] = "2"

    run(args, cfg, path)


if __name__ == "__main__":
    # ENVIRONMENT VARIABLE
    os.environ["WANDB_NOTEBOOK_NAME"] = "test"  # Modify to assign a meaningful name.
    main()
