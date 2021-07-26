from ail.wrapper.action_wrapper import ClipAction, NormalizeAction, RescaleAction

EnvWrapper = {
    "clip_act": ClipAction,
    "normalize_act": NormalizeAction,
    "rescale_act": RescaleAction,
    # "normalize_obs": normalize_obs_v0  # we should implement our own
}
