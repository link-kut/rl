class Environment:
    def __init__(self):
        self.n_states = self.get_n_states()
        self.n_actions = self.get_n_actions()

        self.state_shape = self.get_state_shape()
        self.action_shape = self.get_action_shape()

        self.cnn_input_height = None
        self.cnn_input_width = None
        self.cnn_input_channels = None

        self.continuous = False

    def get_n_states(self):
        pass

    def get_n_actions(self):
        pass

    def get_state_shape(self):
        pass

    def get_action_shape(self):
        pass

    def reset(self):
        pass

    def step(self, action):
        pass

    def close(self):
        pass