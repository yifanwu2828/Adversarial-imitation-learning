from typing import Union, Optional, Tuple, Dict, Any
from collections import OrderedDict

import torch as th
from torch import nn
from torch.cuda.amp import autocast
from torch.distributions import Bernoulli

from ail.agents.irl_agent.irl_core import BaseIRLAgent
from ail.buffer import ReplayBuffer, BufferTag
from ail.common.math import normalize
from ail.common.type_alias import GymSpace, TensorDict
from ail.common.utils import explained_variance
from ail.network.discrim import DiscrimNet


class Adversarial(BaseIRLAgent):
    """
    Base class for adversarial imitation learning algorithms like GAIL and AIRL.

    :param state_space: state space.
    :param action_space: action space.
    :param device: PyTorch device to which the values will be converted.
    :param seed: random seed.
    :param max_grad_norm: Maximum norm for the gradient clipping
    :param epoch_disc: Number of epoch when update the discriminator
    :param replay_batch_size: Replay batch size for training the discriminator
    :param buffer_exp: Replay buffer that store expert demostrations
    :param buffer_kwargs: Arguments to be passed to the buffer.
        eg. : {
            with_reward: False,
            extra_data: ["log_pis"]
            }
    :param gen_algo: RL algorithm for the generator.
    :param gen_kwargs: Kwargs to be passed to the generator.
    :param disc_cls: Class for DiscrimNet,
    :param disc_kwargs: Expected kwargs to be passed to the DiscrimNet.
    :param lr_disc: learning rate for the discriminator
    :param disc_ent_coef: Coefficient for entropy bonus
    :param optim_kwargs: arguments to be passed to the optimizer.
        eg. : {
            "optim_cls": adam,
            "optim_set_to_none": True, # which set grad to None instead of zero.
            }
    :param subtract_logp: wheteher to subtract log_pis from the learning reward.
    :param rew_type: GAIL or AIRL flavor of learning reward.
    :param rew_input_choice: Using logit or logsigmoid or softplus to calculate reward function
    """

    def __init__(
        self,
        state_space: GymSpace,
        action_space: GymSpace,
        device: Union[th.device, str],
        fp16: bool,
        seed: int,
        max_grad_norm: Optional[float],
        epoch_disc: int,
        replay_batch_size: int,
        buffer_exp: Union[ReplayBuffer, str],
        buffer_kwargs: Dict[str, Any],
        gen_algo,
        gen_kwargs: Dict[str, Any],
        disc_cls: DiscrimNet,
        disc_kwargs: Dict[str, Any],
        lr_disc: float,
        disc_ent_coef: float,
        optim_kwargs: Optional[Dict[str, Any]],
        subtract_logp: bool,
        rew_type: str,
        rew_input_choice: str,
        rew_clip: bool,
        max_rew_magnitude: float,
        min_rew_magnitude: float,
        use_absorbing_state: bool,
        infinite_horizon: bool,
        **kwargs,
    ):
        super().__init__(
            state_space,
            action_space,
            device,
            fp16,
            seed,
            max_grad_norm,
            replay_batch_size,
            buffer_exp,
            buffer_kwargs,
            gen_algo,
            gen_kwargs,
            optim_kwargs,
        )
        # DiscrimNet
        self.disc = disc_cls(self.obs_dim, self.act_dim, **disc_kwargs).to(self.device)
        self.lr_disc = lr_disc
        self.optim_disc = self.optim_cls(self.disc.parameters(), lr=self.lr_disc)

        # Lables for the discriminator(Assuming same batch size for gen and exp)
        self.disc_labels = self.make_labels(
            n_gen=self.replay_batch_size, n_exp=self.replay_batch_size
        ).to(self.device)
        self.n_labels = float(len(self.disc_labels))

        # loss function for the discriminator
        self.disc_criterion = nn.BCEWithLogitsLoss(reduction="mean")

        # Coeffient for entropy bonus
        assert disc_ent_coef >= 0, "disc_ent_coef must be non-negative."
        self.disc_ent_coef = disc_ent_coef

        self.learning_steps_disc = 0
        self.epoch_disc = epoch_disc

        # Reward function args
        self.subtract_logp = subtract_logp
        self.rew_type = rew_type
        self.rew_input_choice = rew_input_choice
        self.rew_clip = rew_clip

        if self.rew_clip:
            assert isinstance(max_rew_magnitude, (float, int))
            self.max_rew_magnitude = max_rew_magnitude
            if min_rew_magnitude is None:
                self.min_rew_magnitude = -max_rew_magnitude
            else:
                assert isinstance(min_rew_magnitude, (float, int))
                assert min_rew_magnitude < max_rew_magnitude
                self.min_rew_magnitude = min_rew_magnitude

        self.use_absorbing_state = use_absorbing_state

        if self.use_absorbing_state:
            self.infinite_horizon = infinite_horizon
            absorbing_state = th.zeros(self.state_shape, dtype=th.float32).to(
                self.device
            )
            absorbing_state[-1] = 1.0
            absorbing_action = th.zeros(self.action_shape, dtype=th.float32).to(
                self.device
            )
            absorbing_done = (
                -th.ones(1, dtype=th.float32).reshape(-1, 1).to(self.device)
            )

            self.absorbing_data = {
                "obs": absorbing_state,
                "acts": absorbing_action,
                "next_obs": absorbing_state,
                "dones": absorbing_done,
                "log_pis": None,
                "subtract_logp": False,
            }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"

    def update(
        self, log_this_batch: bool = False
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Main loop
         1. Interact with the environment using the current generator/ policy.
            and store the experience in a replay buffer (implementing in step()).
         2. Update discriminator.
         3. Update generator.
        """
        for _ in range(self.epoch_disc):
            self.learning_steps_disc += 1

            if self.gen.buffer.tag == BufferTag.ROLLOUT:
                # * Sample transitions from ``current`` policy.
                data_gen = self.gen.buffer.sample(self.replay_batch_size)

            elif self.gen.buffer.tag == BufferTag.REPLAY:
                """
                Sampele transition from a ``mixture of all policy``.

                Instead of sampling trajectories from current policy directly,
                we sample transitions from a replay buffer R collected
                while performing off-policy training:
                max_{D} E_R [log(D(s, a))] + E_demo  [log(1 − D(s, a))] − \lambd H(\pi).
                which can be seen as a mixture of all policy distributions that appeared during
                training, instead of the latest trained policy.

                In order to recover the original on-policy expectation,
                one needs to use importance sampling:
                max_{D} E_R [(p_pi/p_R) log(D(s, a))] + E_demo  [log(1 − D(s, a))] − \lambd H(\pi).
                """
                # It can be challenging to properly estimate these densities
                # Algorithm works well in practice with the importance weight omitted.
                data_gen = self.gen.buffer.sample(self.replay_batch_size)

            else:
                raise ValueError(f"Unknown generator buffer type: {self.gen.buffer}.")

            # Samples transitions from expert's demonstrations.
            data_exp = self.buffer_exp.sample(self.replay_batch_size)

            # Calculate log probabilities of generator's actions.
            # And evaluate log probabilities of expert actions.
            # Based on current generator's action distribution.
            # See: https://arxiv.org/pdf/1710.11248.pdf appendix A.2
            if self.subtract_logp:
                with th.no_grad():
                    data_exp["log_pis"] = self.gen.actor.evaluate_log_pi(
                        data_exp["obs"], data_exp["acts"]
                    )
            # Update discriminator.
            disc_logs = self.update_discriminator(data_gen, data_exp, log_this_batch)
            if log_this_batch:
                disc_logs.update(
                    {
                        "lr_disc": self.lr_disc,
                        "learn_steps_disc": self.learning_steps_disc,
                    }
                )
                disc_logs = dict(disc_logs)
            del data_gen, data_exp

        # Calculate rewards:
        if self.gen.buffer.tag == BufferTag.ROLLOUT:
            # Obtain entire batch of transitions from rollout buffer.
            train_policy_data = self.gen.buffer.get()
            # Clear buffer after getting entire buffer.
            self.gen.buffer.reset()

        elif self.gen.buffer.tag == BufferTag.REPLAY:
            # Random uniform sampling a batch of transitions from agent's replay buffer
            train_policy_data = self.gen.buffer.sample(self.gen.batch_size)

        else:
            raise ValueError(f"Unknown generator buffer type: {self.gen.buffer}.")

        # Calculate Batch rewards
        rews = self.disc.calculate_rewards(
            choice=self.rew_input_choice, **train_policy_data
        )

        need_absorbing_return = (
            self.use_absorbing_state and -1 in train_policy_data["dones"]
        )
        if need_absorbing_return:
            # Absorbing state reward
            r_sa = self.disc.calculate_rewards(
                choice=self.rew_input_choice, **self.absorbing_data
            )
            # Final state return keeps the same if absorbing reward is zero.
            if r_sa != 0.0:
                with th.no_grad():
                    rews = self.absorbing_cumulative_return(
                        r_sa,
                        rews,
                        dones=train_policy_data["dones"],
                        remaining_steps=train_policy_data["remaining_steps"],
                        discount=self.gen.gamma,
                        infinite_horizon=self.infinite_horizon,
                    )
        train_policy_data["rews"] = rews

        # Sanity check length of data are equal.
        assert train_policy_data["rews"].shape[0] == train_policy_data["obs"].shape[0]

        # Reward Clipping
        if self.rew_clip:
            train_policy_data["rews"].clamp_(
                self.min_rew_magnitude, self.max_rew_magnitude
            )

        # Update generator using estimated rewards.
        gen_logs = self.update_generator(train_policy_data, log_this_batch)
        if log_this_batch and need_absorbing_return:
            gen_logs.update({"absorbing_rew": r_sa.detach()})
        return gen_logs, disc_logs

    def update_generator(
        self, data: TensorDict, log_this_batch: bool = False
    ) -> Dict[str, Any]:
        """Update generator algo."""
        return self.gen.update_algo(data, log_this_batch)

    def update_discriminator(
        self, data_gen: TensorDict, data_exp: TensorDict, log_this_batch: bool = False
    ) -> Dict[str, Any]:
        """
        Update discriminator.
        Let D denote the probability that a state-action pair (s, a) is classified as expert
        by the discriminator while f is the discriminator logit.

        The objective of the discriminator is to minimize cross-entropy loss
        between expert demonstrations and generated samples:

        L = sum( -E_{exp} [log(D)] - E_{\pi} [log(1 - D)] )

        We write the ``negative`` loss to turn the ``minimization`` problem into ``maximization``.

        -L = sum( E_{exp} [log(D)] + E_{\pi} [log(1 - D)] )

        D = sigmoid(f)
        Output of self.disc() is logits `f` in range (-inf, inf), not [0, 1].
        :param data_gen: batch of data from the current policy
        :param data_exp: batch of data from demonstrations
        """
        self.optim_disc.zero_grad(self.optim_set_to_none)
        with autocast(enabled=self.fp16):
            # Obtain logits of the discriminator.
            disc_logits_gen = self.disc(subtract_logp=self.subtract_logp, **data_gen)
            disc_logits_exp = self.disc(subtract_logp=self.subtract_logp, **data_exp)

            """
            E_{exp} [log(D)] + E_{\pi} [log(1 - D)]
            E_{exp} [log(sigmoid(f))] + E_{\pi} [log(1 - sigmoid(f))]
            *Note: S(x) = 1 - S(-x) -> S(-x) = 1 - S(x)
            
            Implmentation below is correct, but using BCEWithLogitsLoss
            is more numerically stable than using a plain Sigmoid followed by a BCELoss
            
            loss_gen = F.logsigmoid(-logits_gen).mean()
            loss_exp = F.logsigmoid(logits_exp).mean()
            loss_disc = -(loss_gen + loss_exp)
            """
            # Check dimensions of logits.
            assert disc_logits_gen.shape[0] == disc_logits_exp.shape[0]
            # Combine batched logits.
            disc_logits = th.vstack([disc_logits_gen, disc_logits_exp])
            loss_disc = self.disc_criterion(disc_logits, self.disc_labels)

            if self.disc_ent_coef > 0:
                entropy = th.mean(Bernoulli(logits=disc_logits).entropy())
                loss_disc = loss_disc - self.disc_ent_coef * entropy

        self.one_gradient_step(loss_disc, self.optim_disc, self.disc)

        disc_logs = {}
        if log_this_batch:
            # Discriminator's statistics.
            disc_logs = self.compute_disc_stats(
                disc_logits.detach(), loss_disc.detach()
            )
        return disc_logs

    def compute_disc_stats(
        self,
        disc_logits: th.Tensor,
        disc_loss: th.Tensor,
    ) -> Dict[str, float]:
        """
        Train statistics for GAIL/AIRL discriminator, or other binary classifiers.
        :param disc_logits: discriminator logits where expert is 1 and generated is 0
        :param labels: integer labels describing whether logit was for an
                expert (1) or generator (0) sample.
        :param disc_loss: discriminator loss.
        :returns stats: dictionary mapping statistic names for float values.
        """
        with th.no_grad():
            bin_is_exp_pred = disc_logits > 0
            bin_is_exp_true = self.disc_labels > 0
            bin_is_gen_pred = th.logical_not(bin_is_exp_pred)
            bin_is_gen_true = th.logical_not(bin_is_exp_true)

            int_is_exp_pred = bin_is_exp_pred.long()
            int_is_exp_true = bin_is_exp_true.long()
            float_is_gen_pred = bin_is_gen_pred.float()
            float_is_gen_true = bin_is_gen_true.float()

            explained_var_gen = explained_variance(
                float_is_gen_pred.view(-1), float_is_gen_true.view(-1)
            )

            n_exp = float(th.sum(int_is_exp_true))
            n_gen = self.n_labels - n_exp

            percent_gen = n_gen / self.n_labels
            n_gen_pred = int(self.n_labels - th.sum(int_is_exp_pred))

            percent_gen_pred = n_gen_pred / self.n_labels

            correct_vec = th.eq(bin_is_exp_pred, bin_is_exp_true)
            disc_acc = th.mean(correct_vec.float())

            _n_pred_gen = th.sum(th.logical_and(bin_is_gen_true, correct_vec))
            if n_gen < 1:
                gen_acc = float("NaN")
            else:
                # float() is defensive, since we cannot divide Torch tensors by
                # Python ints
                gen_acc = _n_pred_gen / float(n_gen)

            _n_pred_exp = th.sum(th.logical_and(bin_is_exp_true, correct_vec))
            _n_exp_or_1 = max(1, n_exp)
            exp_acc = _n_pred_exp / float(_n_exp_or_1)

            label_dist = Bernoulli(logits=disc_logits)
            entropy = th.mean(label_dist.entropy())
        pairs = [
            ("disc_loss", float(th.mean(disc_loss))),
            # Accuracy, as well as accuracy on *just* expert examples and *just*
            # generated examples
            ("disc_acc", float(disc_acc)),
            ("disc_acc_gen", float(gen_acc)),
            ("disc_acc_exp", float(exp_acc)),
            # Entropy of the predicted label distribution, averaged equally across
            # both classes (if this drops then disc is very good or has given up)
            ("disc_entropy", float(entropy)),
            # True number of generators and predicted number of generators
            ("proportion_gen_true", float(percent_gen)),
            ("proportion_gen_pred", float(percent_gen_pred)),
            ("explained_var_gen", float(explained_var_gen)),
        ]
        return OrderedDict(pairs)

    @staticmethod
    def make_labels(n_gen: int, n_exp: int) -> th.Tensor:
        return th.cat(
            [th.zeros(n_gen, dtype=th.float32), th.ones(n_exp, dtype=th.float32)]
        ).reshape(-1, 1)

    def absorbing_cumulative_return(
        self,
        r_sa: th.Tensor,
        rews: th.Tensor,
        dones: th.Tensor,
        remaining_steps: th.Tensor,
        discount: float,
        infinite_horizon: bool = True,
    ) -> th.Tensor:
        """
        Calculate the cumulative return for the absorbing state.
        The returns for final states are
            R_T = r(s_T, a_T) + sum_{t'=T+1}^{\inf} gamma^t' * r(s_a)
        :param r_sa: the reward for the absorbing state-action pair
        :param rew: rewards of the batch
        :param dones: done signal with absorbing state indicator
        :param remaining_steps: remaining steps in the episode
        :returns: cumulative return.
        """
        # final states before the absorbing state
        done_idx = th.where(dones == 0)[0]
        for i in done_idx:
            if infinite_horizon:
                # In practice, analytical infinite horizon alternative is much less stable.
                rews[i] += discount * r_sa / (1 - discount)
            else:
                # t' = t - T
                t_prime = remaining_steps[i].long()[0]
                power_idx = th.arange(1, t_prime + 1)
                r = r_sa.repeat(t_prime)
                # discount gammas: [gamma, gamma^2,...,gamma^N]
                discounts = th.pow(discount, power_idx).to(self.device)
                # discounts * rewards from T+1 to N or inf
                sum_discounted_rews = th.sum(discounts * r)
                # Final state return
                rews[i] += sum_discounted_rews
        return rews
