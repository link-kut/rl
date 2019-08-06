#import roboschool, gym
import gym

import threading
import time
import json
import os, sys
from environments.environment_names import Environment_Name
from conf.constants_environments import ENVIRONMENT_ID
import paho.mqtt.client as mqtt

GYM_ENV_ID_LIST = [
    Environment_Name.CARTPOLE_V0.value,
    Environment_Name.ROBOSCHOOLANT_V1.value
]

# MQTT Server IP config
MQTT_SERVER = '192.168.0.10'

# MQTT Topic for RIP
MQTT_PUB_TO_SERVO_POWER = 'motor_power'
MQTT_PUB_RESET = 'reset'
MQTT_SUB_FROM_SERVO = 'servo_info'
MQTT_SUB_MOTOR_LIMIT = 'motor_limit_info'
MQTT_SUB_RESET_COMPLETE = 'reset_complete'

STATE_SIZE = 4

balance_motor_power_list = [-60, 0, 60]

env = None
PUB_ID = 0

class Environment:
    def __init__(self, owner):
        if ENVIRONMENT_ID in GYM_ENV_ID_LIST:
            self.env = gym.make(ENVIRONMENT_ID)
        elif ENVIRONMENT_ID == Environment_Name.QUANSER_SERVO_2.value:
            global env
            env = self

            self.episode = 0

            self.state_space_shape = (STATE_SIZE,)
            self.action_space_shape = (len(balance_motor_power_list),)

            self.reward = 0

            self.steps = 0
            self.pendulum_radians = []
            self.state = []
            self.current_pendulum_radian = 0
            self.current_pendulum_velocity = 0
            self.current_motor_velocity = 0

            self.is_swing_up = True
            self.is_state_changed = False
            self.is_motor_limit = False
            self.is_limit_complete = False
            self.is_reset_complete = False

            if owner != "broker":
                self.sub = mqtt.Client(client_id="env_sub", transport="TCP")
                self.sub.on_connect = self.__on_connect
                self.sub.on_message = self.__on_message
                self.sub.username_pw_set(username="link", password="0123")
                self.sub.connect(MQTT_SERVER, 1883, 60)

                sub_thread = threading.Thread(target=self.__sub, args=(self.sub,))
                sub_thread.daemon = True
                sub_thread.start()

                self.pub = mqtt.Client(client_id="env_pub", transport="TCP")
                self.pub.username_pw_set(username="link", password="0123")
                self.pub.connect(MQTT_SERVER, 1883, 60)

        else:
            self.env = None

        self.n_states = self.get_n_states()
        self.n_actions = self.get_n_actions()

    # @staticmethod
    # def __on_connect(client, userdata, flags, rc):
    #     print("mqtt broker connected with result code " + str(rc), flush=False)
    #     client.subscribe(topic=MQTT_SUB_FROM_SERVO)
    #     client.subscribe(topic=MQTT_SUB_MOTOR_LIMIT)
    #     client.subscribe(topic=MQTT_SUB_RESET_COMPLETE)

    ### RIP Define??
    @staticmethod
    def __sub(sub):
        try:
            print("***** Sub thread started!!! *****", flush=False)
            sub.loop_forever()
        except KeyboardInterrupt:
            print("Sub thread KeyboardInterrupted", flush=False)
            sub.unsubscribe(MQTT_SUB_FROM_SERVO)
            sub.unsubscribe(MQTT_SUB_MOTOR_LIMIT)
            sub.unsubscribe(MQTT_SUB_RESET_COMPLETE)
            sub.disconnect()

    def __pub(self, topic, payload, require_response=True):
        global PUB_ID
        self.pub.publish(topic=topic, payload=payload)
        PUB_ID += 1

        if require_response:
            is_sub = False
            while not is_sub:
                if self.is_state_changed or self.is_limit_complete or self.is_reset_complete:
                    is_sub = True
                time.sleep(0.0001)

        self.is_state_changed = False
        self.is_limit_complete = False
        self.is_reset_complete = False

    @staticmethod
    def __on_message(client, userdata, msg):
        global PUB_ID

        if msg.topic == MQTT_SUB_FROM_SERVO:

            servo_info = json.loads(msg.payload.decode("utf-8"))
            motor_radian = float(servo_info["motor_radian"])
            motor_velocity = float(servo_info["motor_velocity"])
            pendulum_radian = float(servo_info["pendulum_radian"])
            pendulum_velocity = float(servo_info["pendulum_velocity"])
            pub_id = servo_info["pub_id"]
            env.__set_state(motor_radian, motor_velocity, pendulum_radian, pendulum_velocity)

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

            env.__set_state(motor_radian, motor_velocity, pendulum_radian, pendulum_velocity)

    def __set_state(self, motor_radian, motor_velocity, pendulum_radian, pendulum_velocity):
        self.is_state_changed = True
        # self.state = [pendulum_radian, pendulum_velocity, motor_radian, motor_velocity]
        self.state = [pendulum_radian, pendulum_velocity]

        self.current_pendulum_radian = pendulum_radian
        self.current_pendulum_velocity = pendulum_velocity
        self.current_motor_velocity = motor_velocity

    def __pendulum_reset(self):
        self.__pub(
            MQTT_PUB_TO_SERVO_POWER,
            "0|pendulum_reset|{0}".format(PUB_ID),
            require_response=False
        )

    # RIP Manual Swing & Balance
    def manual_swingup_balance(self):
        self.__pub(MQTT_PUB_RESET, "reset|{0}".format(PUB_ID))

    # for restarting episode
    def wait(self):
        self.__pub(MQTT_PUB_TO_SERVO_POWER, "0|wait|{0}".format(PUB_ID))

    def get_n_states(self):
        if ENVIRONMENT_ID == Environment_Name.CARTPOLE_V0.value:
            n_states = int(self.env.observation_space.shape[0] / 2)
        else:
            n_states = 2
        return n_states

    def get_n_actions(self):
        if ENVIRONMENT_ID == Environment_Name.ROBOSCHOOLANT_V1.value:
            n_actions = self.env.action_space.shape[0]
        else:
            n_actions = 3
        return n_actions

    def reset(self):
        # state = self.env.reset()
        # if ENVIRONMENT_ID == Environment_Name.CARTPOLE_V0.value:
        #     state = state[2:]
        # else:
        self.steps = 0
        self.pendulum_radians = []
        self.reward = 0
        self.is_motor_limit = False

        wait_time = 1 if self.episode == 0 else 15  # if self.episode % 10 == 0 else 3
        previousTime = time.perf_counter()
        time_done = False

        while not time_done:
            currentTime = time.perf_counter()
            if currentTime - previousTime >= wait_time:
                time_done = True
            time.sleep(0.0001)

        self.__pendulum_reset()
        self.wait()
        self.manual_swingup_balance()
        self.is_motor_limit = False

        self.episode += 1

        return self.state

    def step(self, action):
        # next_state, reward, done, info = self.env.step(action)
        #
        # if ENVIRONMENT_ID == Environment_Name.CARTPOLE_V0.value:
        #     next_state = next_state[2:]
        #     adjusted_reward = reward / 100
        # else:
        motor_power = balance_motor_power_list[action]

        self.__pub(MQTT_PUB_TO_SERVO_POWER, "{0}|{1}|{2}".format(motor_power, "balance", PUB_ID))
        pendulum_radian = self.current_pendulum_radian
        pendulum_angular_velocity = self.current_pendulum_velocity

        next_state = self.state
        self.reward = 1
        adjusted_reward = 1
        self.steps += 1
        self.pendulum_radians.append(pendulum_radian)
        done, info = self.__isDone()

        return next_state, self.reward, adjusted_reward, done, info

    def __isDone(self):
        if self.steps >= 5000:
            return True, "*** Success!!! ***"
        elif self.is_motor_limit:
            self.reward = -100
            return True, "*** Limit position ***"
        elif abs(self.pendulum_radians[-1]) > 3.14 / 24:
            self.is_fail = True
            self.reward = -100
            return True, "*** Fail!!! ***"
        else:
            return False, ""

    def close(self):
        self.pub.publish(topic=MQTT_PUB_TO_SERVO_POWER, payload=str(0))
        self.env.close()
