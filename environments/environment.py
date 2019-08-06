import gym
from environments.environment_names import Environment_Name
from environments.environment_rip import *
import paho.mqtt.client as mqtt
# from gym_unity.envs import UnityEnv

GYM_ENV_ID_LIST = [
    Environment_Name.CARTPOLE_V0.value,
]

ENVIRONMENT_ID = ENVIRONMENT_ID_MINE
ENV_RENDER = ENV_RENDER_MINE
WIN_AND_LEARN_FINISH_SCORE = WIN_AND_LEARN_FINISH_SCORE_MINE
WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES = WIN_AND_LEARN_FINISH_CONTINUOUS_EPISODES_MINE


def get_environment(owner="broker"):
    if ENVIRONMENT_ID == Environment_Name.QUANSER_SERVO_2.value:
        env = Quanser_Servo_2(owner=owner)
    elif ENVIRONMENT_ID == Environment_Name.CARTPOLE_V0.value:
        env = CartPole_v0()
    else:
        env = None
    return env


class Environment:
    def __init__(self):
        self.n_states = self.get_n_states()
        self.n_actions = self.get_n_actions()

        self.state_shape = self.get_state_shape()
        self.action_shape = self.get_action_shape()

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
        self.env = gym.make(ENVIRONMENT_ID)
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


class Quanser_Servo_2(Environment):
    def __init__(self, owner):
        self.env = EnvironmentRIP(owner=owner)
        super(Quanser_Servo_2, self).__init__()
        if owner == "broker":
            self.setup_mqtt_stuff()

    def get_n_states(self):
        n_states = self.env.get_n_actions()
        return n_states

    def get_n_actions(self):
        n_actions = self.env.get_n_actions()
        return n_actions

    def get_state_shape(self):
        return None

    def get_action_shape(self):
        return None

    def reset(self):
        state = self.env.reset()
        return state

    def step(self, action):
        next_state, reward, adjusted_reward, done, info = self.env.step(action)
        return next_state, reward, adjusted_reward, done, info

    def close(self):
        self.env.close()

    def setup_mqtt_stuff(self):

        _env_ = self.env

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
                _env_.__set_state(motor_radian, motor_velocity, pendulum_radian, pendulum_velocity)

            elif msg.topic == MQTT_SUB_MOTOR_LIMIT:
                info = str(msg.payload.decode("utf-8")).split('|')
                pub_id = info[1]
                if info[0] == "limit_position":
                    _env_.is_motor_limit = True
                elif info[0] == "reset_complete":
                    _env_.is_limit_complete = True

            elif msg.topic == MQTT_SUB_RESET_COMPLETE:
                _env_.is_reset_complete = True
                servo_info = str(msg.payload.decode("utf-8")).split('|')
                motor_radian = float(servo_info[0])
                motor_velocity = float(servo_info[1])
                pendulum_radian = float(servo_info[2])
                pendulum_velocity = float(servo_info[3])
                pub_id = servo_info[4]

                _env_.__set_state(motor_radian, motor_velocity, pendulum_radian, pendulum_velocity)

        print("2")

        client = mqtt.Client(client_id="env_sub", transport="TCP")
        client.on_connect = __on_connect
        client.on_message = __on_message
        client.on_log = __on_log
        #client.username_pw_set(username="link", password="0123")
        client.connect(MQTT_SERVER, 1883, 60)

        print("***** Sub thread started!!! *****", flush=False)
        client.loop_start()
        print("3")


