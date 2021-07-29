from ail.wrapper.action_wrapper import (
    ClipBoxAction,
    NormalizeBoxAction,
    RescaleBoxAction,
)

EnvWrapper = {
    "clip_act": ClipBoxAction,
    "normalize_act": NormalizeBoxAction,
    "rescale_act": RescaleBoxAction,
    # "normalize_obs": normalize_obs_v0  # we should implement our own
}
