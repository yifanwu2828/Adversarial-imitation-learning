from functools import partial

from gym.wrappers import FilterObservation, FlattenObservation

from ail.wrapper.action_wrapper import (
    NormalActionNoise,
    ClipBoxAction,
    NormalizeBoxAction,
    RescaleBoxAction,
)
from ail.wrapper.vev_norm_wrapper import VecNormalize
from ail.wrapper.done_on_success import DoneOnSuccessWrapper
from ail.wrapper.absorbing_wrapper import AbsorbingWrapper
from ail.wrapper.time_aware_obs_wrapper import TimeAwareObsWrapper


EnvWrapper = {
    "noisy_act": NormalActionNoise,
    "clip_act": ClipBoxAction,
    "normalize_act": NormalizeBoxAction,
    "rescale_act": RescaleBoxAction,
    "vec_norm": VecNormalize,
    "norm_clip_obs": partial(
        VecNormalize,
        norm_obs=True,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=float("inf"),
    ),
    "norm_clip_rew": partial(
        VecNormalize,
        norm_obs=False,
        norm_reward=True,
        clip_obs=float("inf"),
        clip_reward=10,
    ),
    "done_success": DoneOnSuccessWrapper,
    "absorbing": AbsorbingWrapper,
    "time_aware": partial(TimeAwareObsWrapper, verbose=True),
    "filter_obs": partial(FilterObservation, filter_keys=["desired_goal", "observation"]),
    "flatten_obs": FlattenObservation,
}
