import os
import sys
import shutil
import argparse
from yacs.config import CfgNode as CN


_C = CN()

# TRAINER related parameters
_C.TRAINER = CN()
_C.TRAINER.env_name = "HalfCheetah-v2"  # Name of the experiment
_C.TRAINER.gpu = (0,)  # The gpu ids
_C.TRAINER.logdir = "logs"  # Directory where to write event logs
_C.TRAINER.epochs = 100  # Number of epochs to train
_C.TRAINER.seed = 1  # random seed
_C.TRAINER.scalar_log_freq = 1  # Log scalar parameters every n epochs
_C.TRAINER.video_log_freq = -1  # Log videos every n epochs


# AGENT related parameters
_C.AGENT = CN()
_C.AGENT.reward_update_per_iter = 1  # Number of reward updates per iteration
_C.AGENT.policy_update_per_iter = 1  # Number of policy updates per iteration


# MODEL related parameters
_C.MODEL = CN()
_C.MODEL.n_layers = 2  # Number of hidden layers
_C.MODEL.size = 128  # Width of a fully connected layer
_C.MODEL.output_size = 1  # Width of final layer
_C.MODEL.activation = "tanh"  # Activation function
_C.MODEL.learning_rate = 1e-3  # Optimizer learning rate


# backup the commands
_C.SYS = CN()
_C.SYS.cmds = ""  # Used to backup the commands

FLAGS = _C


def _update_config(FLAGS, args):
    FLAGS.defrost()
    if args.config:
        FLAGS.merge_from_file(args.config)
    if args.opts:
        FLAGS.merge_from_list(args.opts)
    FLAGS.SYS.cmds = " ".join(sys.argv)
    FLAGS.freeze()


def _backup_config(FLAGS, args):
    logdir = FLAGS.TRAINER.logdir
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    # copy the file to logdir
    if args.config:
        shutil.copy2(args.config, logdir)
    # dump all configs
    filename = os.path.join(logdir, "all_configs.yaml")
    with open(filename, "w") as fid:
        fid.write(FLAGS.dump())


def _set_env_var(FLAGS):
    gpus = ",".join([str(a) for a in FLAGS.TRAINER.gpu])
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus


def parse_args(backup=True):
    parser = argparse.ArgumentParser(description="The configs")
    parser.add_argument("--config", help="experiment configure file name", type=str)
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()
    _update_config(FLAGS, args)
    if backup:
        _backup_config(FLAGS, args)
    _set_env_var(FLAGS)
    return FLAGS


if __name__ == "__main__":
    flags = parse_args(backup=True)
    print(flags)
