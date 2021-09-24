import os
from time import time
from datetime import timedelta
from collections import OrderedDict
from typing import Tuple, Dict, Any, Union, Optional

import numpy as np
import torch as th
from tqdm import tqdm

from ail.color_console import COLORS, Console
from ail.common.env_utils import maybe_make_env
from ail.common.pytorch_util import obs_as_tensor
from ail.common.type_alias import GymEnv
from ail.common.utils import set_random_seed, get_stats, countdown


class BaseTrainer:
    """
    Base Class for RL_Trainer and IRL_Trainer.
    :param num_steps: number of steps to train
    :param env: The environment must satisfy the OpenAI Gym API.
    :param env_kwargs: Any kwargs appropriate for the gym env object
        including custom wrapper.
    :param max_ep_len: Total length of a trajectory
        By default, equals to env's own time limit.
    :param eval_interval: How often to evaluate current policy
        By default, we enforce to create a copy of training env for evaluation.
    :param save_freq: How often to save the current policy.
    :param log_dir: path to log directory
    :param log_interval: How often (in terms of steps) to output training info.
        Recommend batch_size * log_every_n_updates
    :param seed: random seed.
    :param verbose: The verbosity level: 0 no output, 1 info, 2 debug.
    :param use_wandb: Wether to use wandb for metrics visualization.
    :param eval_behavior_type:
        stochastic: sample from the distribution (same as behavioral policy used during training),
        mode: use the mode of the Gaussian instead of sampling,
        average: sample half and use mode half and take the average of them.
    """

    __slots__ = [
        "env",
        "env_test",
        "max_ep_len",
        "seed",
        "enable_logging",
        "log_dir",
        "summary_dir",
        "model_dir",
        "use_wandb",
        "writer",
        "tb_tags",
        "save_freq",
        "eval_interval",
        "num_eval_episodes",
        "tochastic_eval_episodes",
        "log_interval",
        "total_timesteps_pbar",
        "best_ret",
        "total_timesteps",
        "verbose",
        "algo",
        "batch_size",
        "device",
        "start_time",
        "use_optuna",
        "global_step",
    ]

    def __init__(
        self,
        total_timesteps: int,
        env: Union[GymEnv, str],
        env_kwargs: Dict[str, Any],
        test_env_kwargs: Dict[str, Any],
        max_ep_len: Optional[int],
        eval_interval: int,
        num_eval_episodes: int,
        save_freq: int,
        log_dir: Optional[str],
        log_interval: int,
        seed: int,
        verbose: int,
        use_wandb: bool,
        eval_behavior_type: str = "mix",
        use_optuna: bool = False,
        **kwargs,
    ):

        if env_kwargs is None:
            # We apply ClipBoxAction wrapper to both env by default
            env_kwargs = {"env_wrapper": ["clip_act"]}

        if test_env_kwargs is None:
            test_env_kwargs = {"env_wrapper": ["clip_act"]}

        # Set RNG seed.
        set_random_seed(seed)

        # Env to collect samples.
        self.env = maybe_make_env(
            env, verbose=verbose, tag="training", color="green", **env_kwargs
        )
        self.seed = seed
        self.env.seed(seed)

        # Env for evaluation.
        if "absorbing" in env_kwargs.get("env_wrapper", []):
            test_env_kwargs["env_wrapper"].append("absorbing")
        self.env_test = maybe_make_env(
            env, verbose=verbose, tag="test", color="magenta", **test_env_kwargs
        )
        self.env_test.seed(2 ** 31 - seed)

        # Set max_ep_len or use default
        self.max_ep_len: int = (
            max_ep_len
            if max_ep_len is not None and isinstance(max_ep_len, int)
            else self.env._max_episode_steps  # noqa
        )

        self.enable_logging = False if log_dir is None or log_dir == "" else True
        if self.enable_logging:
            # Tensorboard/wandb log setting.
            self.log_dir, self.summary_dir, self.model_dir = (
                log_dir,
                os.path.join(log_dir, "summary"),
                os.path.join(log_dir, "model"),
            )

            for d in [self.log_dir, self.summary_dir, self.model_dir]:
                if not os.path.exists(d):
                    os.makedirs(d, exist_ok=True)

        # Check if use wandb
        if use_wandb:
            try:
                import wandb

                self.use_wandb = True
            except ImportError:
                Console.warning(
                    "`wandb` Module Not Found. You can do `pip install wandb` "
                    "If you wish to use it. Fall back to tensorboard logging"
                )
                self.use_wandb = False
        else:
            self.use_wandb = False

        # Placeholder for tensorboard SummaryWriter.
        self.writer = None

        self.tb_tags = {
            "actor_loss": "loss/actor",
            "critic_loss": "loss/critic",
            "entropy_loss": "loss/entropy",
            "entropy": "info/actor/entropy",
            "entropy_coef": "info/actor/entropy_coef",
            "approx_kl": "info/actor/approx_kl",
            "clip_fraction": "info/actor/clip_fraction",
            "disc_loss": "loss/disc",
            "disc_acc": "info/disc/acc_disc",
            "disc_acc_gen": "info/disc/acc_gen",
            "disc_acc_exp": "info/disc/acc_exp",
            "disc_entropy": "info/disc/entropy",
            "proportion_gen_pred": "info/disc/prop_gen",
            "absorbing_rew": "info/disc/absorbing_rew",
        }

        # Log and Saving.
        self.save_freq = save_freq
        self.eval_interval = eval_interval

        self.num_eval_episodes = num_eval_episodes

        eval_behavior_type = eval_behavior_type.lower()
        if eval_behavior_type == "mode":
            self.stochastic_eval_episodes = 0
        elif eval_behavior_type == "average":
            self.stochastic_eval_episodes = num_eval_episodes // 2
        else:
            raise ValueError(
                f"Unrecognized evaluation behavior type: {eval_behavior_type}. "
                f"Valid options are [stochastic, mode, average]."
            )
        self.log_interval = log_interval

        # Progress param
        self.total_timesteps_pbar = tqdm(
            range(1, total_timesteps + 1), dynamic_ncols=True
        )
        self.best_ret = -float("inf")

        # Other parameters.
        self.total_timesteps = total_timesteps
        self.verbose = verbose
        self.algo = None
        self.batch_size = None
        self.device = None
        self.start_time = None
        self.use_optuna = use_optuna
        self._records = []
        self.global_step = 0

    # -----------------------
    # Training/ evaluation
    # -----------------------

    def run_training_loop(self):
        raise NotImplementedError()

    @th.no_grad()
    def evaluate(self, step: int, trial=None) -> None:
        # set algo to evaluation mode
        self.algo.actor.eval()
        train_returns, train_ep_lens = [], []
        valid_returns, valid_ep_lens = [], []
        # visualize result from half explore and half exploit
        for t in range(self.num_eval_episodes):
            obs = self.env_test.reset()
            ep_ret = 0.0
            ep_len = 0
            done = False
            deterministic = False if t < self.stochastic_eval_episodes else True

            while not done:
                obs = obs_as_tensor(obs, self.device)
                if not deterministic:
                    act, _ = self.algo.explore(obs, scale=True)
                else:
                    act = self.algo.exploit(obs, scale=True)
                obs, reward, done, _ = self.env_test.step(act)
                ep_ret += reward
                ep_len += 1

            if deterministic:
                valid_ep_lens.append(ep_len)
                valid_returns.append(ep_ret)
            else:
                train_ep_lens.append(ep_len)
                train_returns.append(ep_ret)

        # Logging evaluation.
        if self.is_eval_logging(step):
            self.eval_logging(
                step,
                train_returns,
                train_ep_lens,
                valid_returns,
                valid_ep_lens,
            )
        # Turn back to train mode.
        self.algo.train()

        mix_returns = np.concatenate([train_returns, valid_returns])
        mean, std = mix_returns.mean(), mix_returns.std()
        if mean > self.best_ret:
            self.best_ret = mean
        Console.info(
            f"Num steps: {step}\t"
            f"| Best Ret: {self.best_ret:.2f}\t"
            f"| Return: {mean:.2f} +/- {std:.2f}"
        )
        if self.use_optuna and trial is not None:
            if len(self._records) == 5:
                self._records.pop(0)
            self._records.append(np.mean(valid_returns))
            # Report intermediate objective value.
            intermediate_value = self.get_records()
            trial.report(intermediate_value, self.global_step)

            # Handle pruning based on the intermediate value.
            if trial.should_prune():
                import optuna

                raise optuna.TrialPruned()

    # -----------------------
    # Logging conditions
    # -----------------------

    def is_train_logging(self, step: int) -> bool:
        return all(
            [
                step % self.log_interval == 0,
                step > self.log_interval,
                self.verbose > 1,
            ]
        )

    def is_eval_logging(self, step: int) -> bool:
        return all(
            [
                step % self.eval_interval == 0,
                step > self.eval_interval,
                self.verbose > 1,
            ]
        )

    def is_saving_model(self, step: int) -> bool:
        cond = (step > 0, step % self.save_freq == 0)
        return all(cond) or step == self.total_timesteps - 1

    def train_logging(
        self, train_logs: Union[Tuple[dict], Dict[str, Any]], step: int
    ) -> None:
        """Log training info (no saving yet)"""
        time_logs = OrderedDict()
        time_logs["total_timestep"] = step
        time_logs["time_elapsed "] = self.duration(self.start_time)

        disc_logging = False
        if len(train_logs) == 2 and isinstance(train_logs, tuple):
            train_logs, disc_logs = train_logs
            disc_logging = True

        train_logs = self.convert_logs(train_logs)

        print("-" * 41)
        self.output_block(train_logs, tag="Train", color="back_bold_green")
        if disc_logging:
            self.output_block(disc_logs, tag="Disc", color="back_bold_red")
        self.output_block(time_logs, tag="Time", color="back_bold_blue")
        print("\n")

    def eval_logging(
        self,
        step: int,
        train_returns: np.ndarray,
        train_ep_lens: np.ndarray,
        eval_returns: np.ndarray,
        eval_ep_lens: np.ndarray,
    ) -> None:
        """Log evaluation info"""
        train_logs, eval_logs, time_logs = OrderedDict(), OrderedDict(), OrderedDict()

        # Time
        time_logs["total_timestep"] = step
        time_logs["time_elapsed "] = self.duration(self.start_time)

        # Train
        if len(train_returns) > 0 and len(train_ep_lens) > 0:
            train_logs["ep_len_mean"] = np.mean(train_ep_lens)
            (
                train_logs["ep_return_mean"],
                train_logs["std_return"],
                train_logs["max_return"],
                train_logs["min_return"],
            ) = get_stats(train_returns)

        # Eval
        eval_logs["ep_len_mean"] = np.mean(eval_ep_lens)
        (
            eval_logs["ep_return_mean"],
            eval_logs["std_return"],
            eval_logs["max_return"],
            eval_logs["min_return"],
        ) = get_stats(eval_returns)
        print("-" * 41)
        self.output_block(train_logs, tag="Train", color="back_bold_green")
        self.output_block(eval_logs, tag="Evaluate", color="back_bold_red")
        self.output_block(time_logs, tag="Time", color="back_bold_blue")
        print("\n")

        if self.enable_logging:
            self.metric_to_tb(step, train_logs, eval_logs)

    @staticmethod
    def output_block(logs: Dict[str, Any], tag: str, color="invisible") -> None:
        """print a block of logs with color and format"""
        print("".join([COLORS[color], f"| {tag + '/': <10}{'|': >29}"]))
        for k, v in logs.items():
            a = f"|  {k: <15}\t{'| '}"
            if isinstance(v, (float, th.Tensor, np.ndarray)):
                b = f"{v: <12.3e}\t|" if abs(v) < 1e-4 else f"{v: <12.4f}\t|"
            else:
                b = f"{v: <12}\t|"
            print("".join([a, b]))
        print("-" * 41)

    # -----------------------
    # Logging/Saving methods
    # -----------------------

    def metric_to_tb(
        self,
        step: int,
        train_logs: Dict[str, Any],
        eval_logs: Dict[str, Any],
    ) -> None:
        # Train logs
        if self.stochastic_eval_episodes > 0:

            self.writer.add_scalar(
                "return/train/ep_len", train_logs.get("ep_len_mean"), step
            )
            self.writer.add_scalar(
                "return/train/ep_rew_mean", train_logs.get("ep_return_mean"), step
            )
            self.writer.add_scalar(
                "return/train/ep_rew_std", train_logs.get("std_return"), step
            )

        # Test log
        self.writer.add_scalar("return/test/ep_len", eval_logs.get("ep_len_mean"), step)
        self.writer.add_scalar(
            "return/test/ep_rew_mean", eval_logs.get("ep_return_mean"), step
        )
        self.writer.add_scalar(
            "return/test/ep_rew_std", eval_logs.get("std_return"), step
        )

    def info_to_tb(self, train_logs: Dict[str, Any], step: int) -> None:
        """Logging to tensorboard or wandb (if sync)"""
        assert train_logs is not None, "train log can not be `None`"
        if len(train_logs) == 2 and isinstance(train_logs, tuple):
            train_logs = {**train_logs[0], **train_logs[1]}

        if len(train_logs) > 0:
            for k, v in train_logs.items():
                if k in self.tb_tags:
                    self.writer.add_scalar(self.tb_tags[k], v, step)

    def save_models(self, save_dir: str, **kwargs) -> None:
        # use algo.sav_mdoels directly for now
        if self.verbose >= 1:
            Console.info(f"Saving model under {save_dir}")
        self.algo.save_models(save_dir)

    # -----------------------
    # Helper functions.
    # -----------------------
    @staticmethod
    def convert_logs(logs: Dict[str, th.Tensor]):
        for k, v in logs.items():
            if isinstance(v, (float, int)):
                continue
            elif isinstance(v, np.ndarray):
                logs[k] = np.mean(v)
            else:
                logs[k] = th.as_tensor(v, dtype=th.float32).mean()
        return logs

    @staticmethod
    def duration(start_time: float) -> str:
        return str(timedelta(seconds=int(time() - start_time)))

    def finish_logging(self) -> None:
        # Wait to ensure that all pending events have been written to disk.
        if self.writer is not None:
            self.writer.flush()
            Console.info(
                "Wait to ensure that all pending events have been written to disk."
            )
            countdown(10)
            self.writer.close()

    def get_records(self):
        return np.mean(self._records)
