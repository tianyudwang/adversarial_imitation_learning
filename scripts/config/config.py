from yacs.config import CfgNode as CN


_C = CN()

# Number of GPU and CPU.
_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1      # Number of GPUS to use in the experiment.
_C.SYSTEM.NUM_WORKERS = -1  # -1 means use all available workers.


# Cuda and AMP settings.
_C.CUDA = CN()
_C.CUDA.fp16 = False   # For FP16 mixed precisiontraining.
_C.CUDA.cudnn = False  # torch.backends.cudnn.


# Optimizers.
_C.OPTIM = CN()
_C.OPTIM.optim_cls = "adam"        # Choices of ["adam", "adamW"].
_C.OPTIM.optim_set_to_none = True  # Set grad to None instead of zero.


# Logging settings.
_C.LOG = CN()
_C.LOG.log_every_n_updates = 20  # Log metrics every n algo updates.
_C.LOG.eval_interval = 5_000     # Perform evaluation every n steps.
_C.LOG.num_eval_episodes = 10    # Number of episodes to run during evaluation.


# Wandb settings. (Mainly for wandb.watch function)
_C.WANDB = CN()
_C.WANDB.log_param = True       # Enable Log in wandb .
_C.WANDB.log_type = "gradients" # Log gradients as histograms in wandb.
_C.WANDB.log_freq = 1000        # Log gradients evey 1000 steps.


# Steps, batch size, buffer size.
_C.STEP = CN()
_C.STEP.num_steps = int(1e6)   # Total number of enviroment steps to run.
_C.STEP.rollout_length = None  # Max episode length if None, it sets to the environment default.


# algo kwargs
_C.ALGO = CN()
_C.ALGO.max_grad_norm = None   # Gradient clipping norm.
_C.ALGO.gamma = 0.99           # Discount factor.
_C.ALGO.seed = 0               # Seed for the random generator.


# PPO only args
_C.PPO = CN()
_C.PPO.epoch_ppo=10       # Number of gradient steps to take during updating.
_C.PPO.clip_eps=0.2       # PPO clipping parameter.
_C.PPO.coef_ent=0.01      # Entropy coefficient for the loss calculation.
_C.PPO.gae_lambda = 0.97  # Lambda parameter for GAE

_C.PPO.batch_size = 1_000 # Rollout buffer batch size to use during updating.
_C.PPO.with_reward = True # Wether to store reward in rollout buffer.
_C.PPO.extra_data=["log_pis"]  # Extra data to store in rollout buffer.

# PPO policy_kwargs
_C.PPO.pi = (64, 64)     # Policy hidden layers & sizes.
_C.PPO.vf = (64, 64)     # Value function hidden layers & sizes.
_C.PPO.activation = "relu_inplace" # Hidden activation.
_C.PPO.critic_type="V"   # Critic type V: value function.
_C.PPO.lr_actor = 3e-4   # Policy learning rate.
_C.PPO.lr_critic = 3e-4  # Value function learning rate.

# SAC only args
_C.SAC = CN()
_C.SAC.start_steps = 10_000    # Number of exploration steps without updates.
_C.SAC.num_gradient_steps = 1  # Number of gradient steps to take per algo update.
_C.SAC.target_update_interval = 1  # Number of traget updates on per algo update.
_C.SAC.tau = 0.005             # Soft update coefficient. 
_C.SAC.log_alpha_init = 1.0    # Init value of log_alpha.
_C.SAC.lr_alpha = 3e-4         # Learning rate for log_alpha.

# SAC 
_C.SAC.batch_size = 256       # Replay buffer batch size to use during updating.
_C.SAC.buffer_size = int(1e6) # Capactiy of the replay buffer.
_C.SAC.with_reward = True     # Wether to store reward in replay buffer.
_C.SAC.extra_data=[]          # Extra data to store in replay buffer.

# SAC policy_kwargs
_C.SAC.pi = (128, 128)    # Policy hidden layers & sizes.
_C.SAC.qf = (128, 128)    # Q-function hidden layers & sizes.
_C.SAC.activation = "relu_inplace" # Hidden activation
_C.SAC.critic_type="twin"     # Critic type twin: (double Q).
_C.SAC.lr_actor = 7.3 * 1e-4  # Policy learning rate.
_C.SAC.lr_critic =7.3 * 1e-4  # Q-function learning rate.

# Discriminator settings
_C.DISC = CN()
_C.DISC.spectral_norm = False  # Apply Spectral Norm.
_C.DISC.dropout = False        # Enable dropout.

# Discriminator Architecture
_C.DISC.hidden_units = (128, 128)  # Discriminator hidden layers & sizes.
_C.DISC.hidden_activation = "relu_inplace" # Hidden activation.

_C.DISC.epoch_disc = 1  # Update discriminator n times per update.
_C.DISC.lr_disc = 3e-4  # Discriminator learning rate.


# AIRL only args
_C.AIRL = CN()
_C.AIRL.disc_cls = "airl_sa"   # Choices = ["airl_so", "airl_sa"].
_C.AIRL.subtract_logp = True   # Subtract log_p from discrtiminator logits.
# Reward formulation
_C.AIRL.rew_type = "airl"      # Choices = ["airl", "gail"].
_C.AIRL.rew_input_choice = "logit" # Choices = ["logsigmoid", "softplus", "logti"].


def get_cfg_defaults():
  """Get a yacs CfgNode object with default values for my_project."""
  # Return a clone so that the defaults will not be altered
  # This is for the "local variable" use pattern
  return _C.clone()

