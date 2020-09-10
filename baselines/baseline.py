from stable_baselines.deepq.policies import MlpPolicy, CnnPolicy
from stable_baselines import DQN
import argparse

from custom_callback import CustomCallback
from wrappers import make_atari, wrap_atari_dqn
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--learning_rate', type=float, default=0.00025)
parser.add_argument('--train_freq', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_starts', type=int, default=50000)
parser.add_argument('--target_net_update_freq', type=int, default=10000)
parser.add_argument('--epsilon_max', type=float, default=1.0)
parser.add_argument('--epsilon_min', type=float, default=0.01)
parser.add_argument('--epsilon_decay_end_step', type=int, default=1000000)
parser.add_argument('--replay_memory_capacity', type=int, default=250000)
parser.add_argument('--max_steps', type=int, default=5000000)
parser.add_argument('--max_runs', type=int, default=1)
parser.add_argument('--data_save_step_interval', type=int, default=20000)
parser.add_argument('--n_step', type=bool, default=False)
parser.add_argument('--experiments_n_steps', type=list, default=[6])  # 1, 3, 6, 'Omega'
parser.add_argument('--omega_step_window_size', type=int, default=6)
parser.add_argument('--net_model', type=str, default='CNN',
                    help="Select CNN or MLP for DQN network_model")
parser.add_argument('--continue_training', type=bool, default=False)
# Environment Arguments
# parser.add_argument('--env', type=str, default='PongDeterministic-v4',
#                     help='Environment Name')  # PongDeterministic-v4, PongNoFrameskip-v4, BreakoutDeterministic-v4
parser.add_argument('--env', type=str, default='PongNoFrameskip-v4',
                    help='Environment Name')  # PongDeterministic-v4, PongNoFrameskip-v4, BreakoutDeterministic-v4
parser.add_argument('--episode-life', type=int, default=1,
                    help='Whether env has episode life(1) or not(0)')
parser.add_argument('--clip-rewards', type=int, default=1,
                    help='Whether env clip rewards(1) or not(0)')
parser.add_argument('--frame-stack', type=int, default=1,
                    help='Whether env stacks frame(1) or not(0)')
parser.add_argument('--scale', type=int, default=1,
                    help='Whether env scales(1) or not(0)')

args = parser.parse_args()


def print_args():
    print('\n' + 'Argument Options')
    for k, v in vars(args).items():
        print(k + ': ' + str(v))
    print()

print_args()

env = make_atari(args.env)
env = wrap_atari_dqn(env, args)

dqn = DQN(
    CnnPolicy, env,
    buffer_size=args.replay_memory_capacity,
    exploration_fraction=0.2,
    exploration_final_eps=args.epsilon_min,
    exploration_initial_eps=args.epsilon_max,
    train_freq=args.train_freq,
    batch_size=args.batch_size * args.train_freq,
    double_q=True,
    learning_starts=args.learning_starts,
    target_network_update_freq=args.target_net_update_freq,
    prioritized_replay=True,
    prioritized_replay_alpha=0.6,
    prioritized_replay_beta0=0.4,
    prioritized_replay_beta_iters=None,
    prioritized_replay_eps=0.001,
    param_noise=False,
    verbose=2
)

callback = CustomCallback(max_steps=args.max_steps, dqn=dqn)

dqn.learn(callback=callback, total_timesteps=args.max_steps)
dqn.save("deepq_breakout")

del dqn # remove to demonstrate saving and loading

dqn = DQN.load("deepq_breakout")

obs = env.reset()
while True:
    action, _states = dqn.predict(obs)
    obs, rewards, dones, info = env.step(action)
    env.render()