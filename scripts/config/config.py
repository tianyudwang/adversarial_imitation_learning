from yacs.config import CfgNode as CN

import pathlib

_C = CN()

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1    # Number of GPUS to use in the experiment
_C.SYSTEM.NUM_WORKERS = -1  # -1 means use all available workers


# Cuda and AMP settings
_C.CUDA = CN()
_C.CUDA.cuda = False
_C.CUDA.fp16 = False
_C.CUDA.cudnn = False


# Optimizers
_C.OPTIM = CN()
_C.OPTIM.optim_cls = 'adam'
_C.OPTIM.optim_set_to_none = True


# Logging settings
_C.LOG = CN()
_C.LOG.log_every_n_updates = 20
_C.LOG.eval_interval = 5_000
_C.LOG.num_eval_episodes = 10


# Wandb settings (Mainly for wandb.watch function)
_C.WANDB = CN()
_C.WANDB.log_param = True
_C.WANDB.log_type = "gradients"
_C.WANDB.log_freq = 1000


# Miscellaneous
_C.MISC = CN()
_C.MISC.seed = 0
_C.MISC.verbose = 2


# Steps, batch size, buffer size
_C.STEP = CN()
_C.STEP.num_steps = int(1e6)
_C.STEP.rollout_length = None


# Buffer
_C.BUFFER = CN()
_C.BUFFER.batch_size = 256
_C.BUFFER.buffer_size = int(1e6)

# algo kwargs
_C.ALGO = CN()
_C.ALGO.max_grad_norm = None
_C.ALGO.gamma = 0.99

# PPO only args
_C.PPO = CN()
_C.PPO.epoch_ppo=10
_C.PPO.gae_lambda=0.97
_C.PPO.clip_eps=0.2
_C.PPO.coef_ent=0.01

_C.PPO.pi = (64, 64)
_C.PPO.vf = (64, 64)
_C.PPO.activation = 'relu'
_C.PPO.lr_actor = 3e-4
_C.PPO.lr_critic = 3e-4

# SAC only args
_C.SAC = CN()
_C.SAC.start_steps = 10_000
_C.SAC.num_gradient_steps = 1  # ! slow O(n)
_C.SAC.target_update_interval = 1  # Recommend to sync with num_gradient_steps.
_C.SAC.tau = 0.005
_C.SAC.log_alpha_init = 1.0
_C.SAC.lr_alpha = 3e-4

_C.SAC.pi = (128, 128)
_C.SAC.qf = (128, 128)
_C.SAC.activation = 'relu'
_C.SAC.lr_actor = 7.3 * 1e-4
_C.SAC.lr_critic =7.3 * 1e-4

def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()


