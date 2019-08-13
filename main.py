import glob
import time
from multiprocessing import Process

import sys, os
idx = os.getcwd().index("{0}rl".format(os.sep))
PROJECT_HOME = os.getcwd()[:idx+1] + "rl{0}".format(os.sep)
sys.path.append(PROJECT_HOME)

from conf.constants_general import PYTHON_PATH, NUM_WORKERS

if not os.path.exists(os.path.join(PROJECT_HOME, "graphs")):
    os.makedirs(os.path.join(PROJECT_HOME, "graphs"))

if not os.path.exists(os.path.join(PROJECT_HOME, "logs")):
    os.makedirs(os.path.join(PROJECT_HOME, "logs"))

if not os.path.exists(os.path.join(PROJECT_HOME, "out_err")):
    os.makedirs(os.path.join(PROJECT_HOME, "out_err"))

if not os.path.exists(os.path.join(PROJECT_HOME, "models", "model_save_files")):
    os.makedirs(os.path.join(PROJECT_HOME, "models", "model_save_files"))


def run_chief():
    try:
        os.system(PYTHON_PATH + " " + os.path.join(PROJECT_HOME, "chief.py"))
        sys.stdout = open(os.path.join(PROJECT_HOME, "out_err", "chief_stdout.out"), "wb")
        sys.stderr = open(os.path.join(PROJECT_HOME, "out_err", "chief_stderr.out"), "wb")
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stderr.flush()


def run_worker(worker_id):
    try:
        os.system(PYTHON_PATH + " " + os.path.join(PROJECT_HOME, "worker.py") + " {0}".format(worker_id))
        sys.stdout = open(os.path.join(PROJECT_HOME, "out_err", "worker_{0}_stdout.out").format(worker_id), "wb")
        sys.stderr = open(os.path.join(PROJECT_HOME, "out_err", "worker_{0}_stderr.out").format(worker_id), "wb")
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    # logger = multiprocessing.log_to_stderr()
    # logger.setLevel(logging.INFO)
    # m = multiprocessing.Manager()

    response = input("DELETE All Graphs and Logs Files? [y/n]: ")

    if response == "Y" or response == "y":
        files = glob.glob(os.path.join(PROJECT_HOME, "graphs", "*"))
        for f in files:
            os.remove(f)

        files = glob.glob(os.path.join(PROJECT_HOME, "logs", "*"))
        for f in files:
            os.remove(f)

        files = glob.glob(os.path.join(PROJECT_HOME, "out_err", "*"))
        for f in files:
            os.remove(f)

        files = glob.glob(os.path.join(PROJECT_HOME, "models", "model_save_files", "*"))
        for f in files:
            os.remove(f)

        chief = Process(target=run_chief, args=())
        chief.start()

        time.sleep(1)

        workers = []
        for worker_id in range(NUM_WORKERS):
            worker = Process(target=run_worker, args=(worker_id,))
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()

        chief.join()
