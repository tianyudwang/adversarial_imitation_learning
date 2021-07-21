from ail.agents.irl_agent.irl_core import BaseIRLAgent


class AIRL(BaseIRLAgent):
    def __init__(self, agent_flags):
        super(AIRL, self).__init__()

        self.flags = agent_flags

        # Discriminator
        # self.discriminator = MLP

        # Generator
        # self.generator = PPO/SAC ...

    def train_discriminator(self):
        pass

    def train_generator(self):
        pass
