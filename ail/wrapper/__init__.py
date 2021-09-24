from functools import partial

from ail.wrapper.action_wrapper import (
    NormalActionNoise,
    ClipBoxAction,
    NormalizeBoxAction,
    RescaleBoxAction,
)

from ail.wrapper.vev_norm_wrapper import VecNormalize
from ail.wrapper.absorbing_wrapper import AbsorbingWrapper


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
    "absorbing": AbsorbingWrapper,
}
