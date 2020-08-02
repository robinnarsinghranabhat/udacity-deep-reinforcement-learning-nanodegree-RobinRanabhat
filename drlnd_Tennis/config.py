class Config:
    """Configuration class for parameters"""
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method."""
        if Config.__instance is None:
            Config()
        return Config.__instance

    def __init__(self, seed=None, device=None, target_score=None,
                 target_episodes=None, max_episodes=None, num_agents=None,
                 action_size=None, state_size=None, lr_actor=None,
                 lr_critic=None, weight_decay=None, buffer_size=None,
                 batch_size=None, gamma=None, tau=None, update_every=None,
                 intial_noise_decay_parameter=None, noise_reduction=None):
        """ Virtually private constructor.
        Initialize an Config object.
        Params
        ======
            seed (int): random seed
            device (string): cuda/cpu
            target_episodes (int): number of episodes in which target is to be reached
            target_score (float): target to reach in target_episodes consecutive episodes
            max_episodes (int): maximum number of episodes used in training
            print_every (int): print output in every print_every episodes
            num_agents (int): number of agents in the environment
            action_size (int): dimension of each action
            state_size (int): dimension of each state
            lr_actor (float): learning rate of the actor
            lr_critic (float): learning rate of the critic
            weight_decay (float): L2 weight decay
            buffer_size (int): replay buffer size
            batch_size (int): minibatch size
            gamma (float): discount factor
            tau (float): for soft update of target parameters
            update_every (int): how often to update the network
            intial_noise_decay_parameter (float): amplitude of OU noise
            noise_reduction (float): noise decay rate
        """
        Config.__instance = self

        self.seed = None
        self.device = None

        self.target_score_to_beat = None
        self.target_episodes = None
        self.max_episodes = None
        self.print_point = None

        self.num_agents = None
        self.action_size = None
        self.state_size = None

        self.lr_actor = None
        self.lr_critic = None
        self.weight_decay = None
        self.buffer_size = None
        self.batch_size = None
        self.gamma = None
        self.tau = None
        self.update_every = None

        self.starting_noise = None
        self.noise_reduction = None
