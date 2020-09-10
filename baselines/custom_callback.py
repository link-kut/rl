from stable_baselines.common.callbacks import BaseCallback
import time

class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: (int) Verbosity level 0: not output 1: info 2: debug
    """
    def __init__(self, max_steps, dqn, verbose=0):
        super(CustomCallback, self).__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseRLModel
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # type: logger.Logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]
        self.episode = 0
        self.max_steps = max_steps
        self.dqn = dqn
        self.train_start_time = 0.0
        self.current_steps = 0
        self.episode_steps = 0

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        self.current_steps = 0
        self.episode_start_time = time.time()
        self.episode_start_steps = self.current_steps
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: (bool) If the callback returns False, training is aborted early.
        """
        self.current_steps += 1

        if len(self.locals['episode_rewards']) != self.episode:
            if self.episode != 0:
                frame_per_sec = int((self.current_steps - self.episode_steps) / (time.time() - self.episode_start_time))
                print("Episode: {0:4}, global_timesteps: {1:8}, Episode Reward: {2:5}, FPS: {3:4}".format(
                    self.episode,
                    self.num_timesteps,
                    self.locals['episode_rewards'][-2],
                    frame_per_sec
                ))
            self.episode += 1
            self.episode_start_time = time.time()
            self.episode_steps = self.current_steps

        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass