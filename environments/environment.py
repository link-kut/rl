# https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
import gym

# from gym_unity.envs import UnityEnv

from conf.constants_mine import MQTT_SERVER_FOR_RIP
from environments.envs.environment_rip import *
from conf.names import *
import paho.mqtt.client as mqtt
from gym_unity.envs import UnityEnv

GYM_ENV_ID_LIST = [
    EnvironmentName.CARTPOLE_V0.value,
]

ENV_RENDER = ENV_RENDER_MINE
WIN_AND_LEARN_FINISH_SCORE = WIN_AND_LEARN_FINISH_SCORE_MINE
WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES = WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES_MINE


def get_environment(owner="chief"):
    if ENVIRONMENT_ID == EnvironmentName.QUANSER_SERVO_2:
        client = mqtt.Client(client_id="env_sub_2", transport="TCP")
        env = EnvironmentRIP(mqtt_client=client)

        def __on_connect(client, userdata, flags, rc):
            print("mqtt broker connected with result code " + str(rc), flush=False)
            client.subscribe(topic=MQTT_SUB_FROM_SERVO)
            client.subscribe(topic=MQTT_SUB_MOTOR_LIMIT)
            client.subscribe(topic=MQTT_SUB_RESET_COMPLETE)

        def __on_log(client, userdata, level, buf):
            print(buf)

        def __on_message(client, userdata, msg):
            global PUB_ID

            if msg.topic == MQTT_SUB_FROM_SERVO:
                servo_info = json.loads(msg.payload.decode("utf-8"))
                motor_radian = float(servo_info["motor_radian"])
                motor_velocity = float(servo_info["motor_velocity"])
                pendulum_radian = float(servo_info["pendulum_radian"])
                pendulum_velocity = float(servo_info["pendulum_velocity"])
                pub_id = servo_info["pub_id"]
                env.set_state(motor_radian, motor_velocity, pendulum_radian, pendulum_velocity)

            elif msg.topic == MQTT_SUB_MOTOR_LIMIT:
                info = str(msg.payload.decode("utf-8")).split('|')
                pub_id = info[1]
                if info[0] == "limit_position":
                    env.is_motor_limit = True
                elif info[0] == "reset_complete":
                    env.is_limit_complete = True

            elif msg.topic == MQTT_SUB_RESET_COMPLETE:
                env.is_reset_complete = True
                servo_info = str(msg.payload.decode("utf-8")).split('|')
                motor_radian = float(servo_info[0])
                motor_velocity = float(servo_info[1])
                pendulum_radian = float(servo_info[2])
                pendulum_velocity = float(servo_info[3])
                pub_id = servo_info[4]
                env.set_state(motor_radian, motor_velocity, pendulum_radian, pendulum_velocity)


        client.on_connect = __on_connect
        client.on_message =  __on_message
        client.on_log = __on_log

        # client.username_pw_set(username="link", password="0123")
        client.connect(MQTT_SERVER_FOR_RIP, 1883, 60)

        print("***** Sub thread started!!! *****", flush=False)
        client.loop_start()

    elif ENVIRONMENT_ID == EnvironmentName.CARTPOLE_V0:
        env = CartPole_v0()
    elif ENVIRONMENT_ID == EnvironmentName.CHASER_V1:
        env = Chaser_v1()
    elif ENVIRONMENT_ID == EnvironmentName.BREAKOUT_DETERMINISTIC_V4:
        env = BreakoutDeterministic_v4()
    else:
        env = None
    return env


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


class CartPole_v0(Environment):
    def __init__(self):
        self.env = gym.make(ENVIRONMENT_ID.value)
        super(CartPole_v0, self).__init__()

    def get_n_states(self):
        n_states = int(self.env.observation_space.shape[0] / 2)
        return n_states

    def get_n_actions(self):
        n_actions = self.env.action_space.n
        return n_actions

    def get_state_shape(self):
        state_shape = list(self.env.observation_space.shape)
        state_shape[0] = int(state_shape[0] / 2)
        return tuple(state_shape)

    def get_action_shape(self):
        action_shape = self.env.action_space.shape
        return action_shape

    def reset(self):
        state = self.env.reset()
        return state[2:]

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        next_state = next_state[2:]
        adjusted_reward = reward / 100

        return next_state, reward, adjusted_reward, done, info

    def close(self):
        self.env.close()


class Chaser_v1(Environment):
    unity_env_worker_id = 0

    def __init__(self):
        self.env = UnityEnv(
            environment_filename=ENVIRONMENT_ID.value,
            worker_id=Chaser_v1.unity_env_worker_id,
            use_visual=True,
            multiagent=True
        ).unwrapped
        self.increase_env_worker_id()
        super(Chaser_v1, self).__init__()
        self.action_shape = self.get_action_shape()
        self.state_shape = self.get_state_shape()

        self.cnn_input_height = self.state_shape[0]
        self.cnn_input_width = self.state_shape[1]
        self.cnn_input_channels = self.state_shape[2]

        self.continuous = True

    def increase_env_worker_id(self):
        Chaser_v1.unity_env_worker_id += 1

    def get_n_states(self):
        n_states = 3
        return n_states

    def get_n_actions(self):
        n_actions = 3
        return n_actions

    def get_state_shape(self):
        return self.env.observation_space.shape

    def get_action_shape(self):
        return self.env.action_space.shape

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        adjusted_reward = reward

        return next_state, reward, adjusted_reward, done, info

    def close(self):
        self.env.close()


class BreakoutDeterministic_v4(Environment):
    def __init__(self):
        self.env = gym.make(ENVIRONMENT_ID.value)
        super(BreakoutDeterministic_v4, self).__init__()
        self.action_shape = self.get_action_shape()
        self.state_shape = self.get_state_shape()
        self.cnn_input_height = self.state_shape[0]
        self.cnn_input_width = self.state_shape[1]
        self.cnn_input_channels = self.state_shape[2]

    def to_grayscale(self, img):
        return np.mean(img, axis=2)

    def downsample(self, img):
        return img[::2, ::2]

    def preprocess(self, img):
        gray_frame = self.to_grayscale(self.downsample(img))
        gray_frame = np.expand_dims(gray_frame, axis=0)
        return gray_frame

    def transform_reward(self, reward):
        return np.sign(reward)

    def get_n_states(self):
        return None

    def get_n_actions(self):
        return self.env.action_space.n

    def get_state_shape(self):
        state_shape = (int(self.env.observation_space.shape[0]/2), int(self.env.observation_space.shape[1]/2), 1)
        return state_shape

    def get_action_shape(self):
        action_shape = self.env.action_space.n
        return (action_shape,)

    def reset(self):
        state = self.env.reset()
        return self.preprocess(state)

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        adjusted_reward = self.transform_reward(reward)

        return self.preprocess(next_state), reward, adjusted_reward, done, info

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()


if __name__ == "__main__":
    env = get_environment()
    # Reset it, returns the starting frame
    frame = env.reset()
    print(env.get_state_shape())
    print(env.get_action_shape())
    print(frame.shape)

    # Render
    env.render()

    is_done = False
    last_frame = frame

    idx = 0
    while not is_done:
        # Perform a random action, returns the new frame, reward and whether the game is over
        frame, reward, adjusted_reward, is_done, _ = env.step(env.action_space.sample())

        state = frame - last_frame

        print(idx, state.mean(), reward, adjusted_reward, is_done)

        last_frame = frame
        idx = idx + 1

        # Render
        env.render()
