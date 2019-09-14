import sys, os

idx = os.getcwd().index("{0}rl".format(os.sep))
PROJECT_HOME = os.getcwd()[:idx+1] + "rl{0}".format(os.sep)
sys.path.append(PROJECT_HOME)

from rl_main import rl_utils

env = rl_utils.get_environment()

if __name__ == "__main__":
    algorithm = rl_utils.get_rl_algorithm(env)
    policy = algorithm.start_iteration()

    print(policy)
