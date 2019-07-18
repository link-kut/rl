import pybullet as p
import time
import pybullet_data
import pybullet_envs.agents.train_ppo
import pybullet_envs
import gym
env = gym.make("AntBulletEnv-v0")

env.render()

state = env.reset()
print(state)

while True:
    pass
