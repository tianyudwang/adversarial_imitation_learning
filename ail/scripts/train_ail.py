import os

import numpy as np
import torch
import gym

import ail.common.pytorch_util as ptu

from ail.agents.irl_agent.airl_agent import AIRL
from config import parse_args


class AIL_Trainer(object):
    def __init__(self, flags):

        #############
        ## INIT
        #############

        # Get flags, create logger
        self.flags = flags.TRAINER

        # Set random seeds
        self.seed = self.flags.seed
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        ptu.init_gpu(use_gpu=len(self.flags.gpu) > 0, gpu_id=self.flags.gpu[0])

        #############
        # ENV and AGENT
        #############

        self.init_env()
        self.init_agent(flags.AGENT)

    def init_env(self):
        """
        Initialize gym environment and create env wrapper if necessary
        """
        self.env = gym.make(self.flags.env_name)
        self.env.seed(self.seed)

        # Find maximum episode length and set to video length

    def init_agent(self, flags):
        """
        Initialize agent
        """

        # Observation and action sizes
        ob_dim = self.env.observation_space.shape[0]
        ac_dim = self.env.action_space.shape[0]
        flags.defrost()
        flags.ob_dim = ob_dim
        flags.ac_dim = ac_dim
        flags.freeze()
        self.agent = AIRL(flags)

    def run_training_loop(self):
        """ """
        pass

        # Obtain expert trajectories

        # Main loop
        # 1. Interact with the environment using the current generator
        #    and store the experience in a replay buffer
        # 2. Update discriminator
        # 3. Update generator


if __name__ == "__main__":
    # Load configs
    FLAGS = parse_args()

    # Create directory for logging

    trainer = AIL_Trainer(FLAGS)
    trainer.run_training_loop()
