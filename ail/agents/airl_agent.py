from ail.agents.base_agent import BaseAgent


class AIRL_Agent(BaseAgent):
    def __init__(self, agent_flags):
        super(AIRL_Agent, self).__init__()

        self.flags = agent_flags

        # Discriminator
        # self.discriminator = MLP

        # Generator
        # self.generator = PPO/SAC ...

    def train_discriminator(self):
        pass

    def train_generator(self):
        pass
