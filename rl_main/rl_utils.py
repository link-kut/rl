import json

import paho.mqtt.client as mqtt
import torch

from rl_main.conf.constants_mine import *
from rl_main.environments.gym.breakout import BreakoutDeterministic_v4
from rl_main.environments.gym.cartpole import CartPole_v0
from rl_main.environments.gym.pendulum import Pendulum_v0
from rl_main.environments.real_device.environment_rip import EnvironmentRIP
from rl_main.environments.unity.chaser_unity import Chaser_v1
from rl_main.environments.unity.drone_racing import Drone_Racing
from rl_main.models.actor_critic_cnn import ActorCriticCNN
from rl_main.models.actor_critic_mlp import ActorCriticMLP
from rl_main.models.cnn import CNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


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
    elif ENVIRONMENT_ID == EnvironmentName.PENDULUM_V0:
        env = Pendulum_v0(MY_PLATFORM)
    elif ENVIRONMENT_ID == EnvironmentName.DRONE_RACING:
        env = Drone_Racing(MY_PLATFORM)
    else:
        env = None
    return env


def get_rl_model(env):
    if DEEP_LEARNING_MODEL == ModelName.ActorCriticMLP:
        model = ActorCriticMLP(
            s_size=env.n_states,
            a_size=env.n_actions,
            continuous=env.continuous,
            device=device
        ).to(device)
    elif DEEP_LEARNING_MODEL == ModelName.ActorCriticCNN:
        model = ActorCriticCNN(
            input_height=env.cnn_input_height,
            input_width=env.cnn_input_width,
            a_size=env.n_actions,
            continuous=env.continuous,
            device=device
        ).to(device)
    elif DEEP_LEARNING_MODEL == ModelName.CNN:
        model = CNN(
            input_height=env.cnn_input_height,
            input_width=env.cnn_input_width,
            input_channels=env.cnn_input_channels,
            a_size=env.n_actions,
            continuous=env.continuous,
            device=device
        ).to(device)
    else:
        model = None
    return model