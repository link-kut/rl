import threading
import time
import json
import numpy as np
from conf.constants_mine import *

# MQTT Server IP config
MQTT_SERVER = MQTT_SERVER_MINE

# MQTT Topic for RIP
MQTT_PUB_TO_SERVO_POWER = 'motor_power'
MQTT_PUB_RESET = 'reset'
MQTT_SUB_FROM_SERVO = 'servo_info'
MQTT_SUB_MOTOR_LIMIT = 'motor_limit_info'
MQTT_SUB_RESET_COMPLETE = 'reset_complete'

STATE_SIZE = 4

balance_motor_power_list = [-60, 0, 60]

PUB_ID = 0


class EnvironmentRIP:
    def __init__(self, owner):
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
        self.previous_time = 0.0

        self.is_swing_up = True
        self.is_state_changed = False
        self.is_motor_limit = False
        self.is_limit_complete = False
        self.is_reset_complete = False

    def __pub(self, topic, payload, require_response=True):
        global PUB_ID
        self.sub.publish(topic=topic, payload=payload)
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
        n_states = 2
        return n_states

    def get_n_actions(self):
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
        self.previous_time = time.perf_counter()

        return np.asarray(self.state)

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

        next_state = np.asarray(self.state)
        self.reward = 1.0
        adjusted_reward = 1.0
        self.steps += 1
        self.pendulum_radians.append(pendulum_radian)
        done, info = self.__isDone()

        if not done:
            while True:
                current_time = time.perf_counter()
                if current_time - self.previous_time >= 6 / 1000:
                    break
        else:
            self.wait()

        self.previous_time = time.perf_counter()

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
        # self.env.close()