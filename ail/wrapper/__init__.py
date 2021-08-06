from ail.wrapper.action_wrapper import (
    ClipBoxAction,
    NormalizeBoxAction,
    RescaleBoxAction,
)
from ail.wrapper.vec_norm_wrapper import VecNormalize

EnvWrapper = {
    "clip_act": ClipBoxAction,
    "normalize_act": NormalizeBoxAction,
    "rescale_act": RescaleBoxAction,
    "vec_norm": VecNormalize,
}
