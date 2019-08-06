import os, sys
import glob
import time
from multiprocessing import Process
from conf.constants_general import PYTHON_PATH, NUM_WORKERS

if not os.path.exists("./logs/"):
    os.makedirs("./logs/")

if not os.path.exists("./out_err/"):
    os.makedirs("./out_err/")


def run_worker(worker_id):
    try:
        os.system(PYTHON_PATH + " ./worker.py {0}".format(worker_id))
        sys.stdout = open("./out_err/worker_{0}_stdout.out".format(worker_id), "wb")
        sys.stderr = open("./out_err/worker_{0}_stderr.out".format(worker_id), "wb")
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stderr.flush()


if __name__ == "__main__":
    # logger = multiprocessing.log_to_stderr()
    # logger.setLevel(logging.INFO)
    # m = multiprocessing.Manager()

    response = input("DELETE All Graphs and Logs Files? [y/n]: ")

    if response == "Y" or response == "y":
        files = glob.glob('./logs/*')
        for f in files:
            os.remove(f)

        files = glob.glob('./out_err/*')
        for f in files:
            os.remove(f)

    worker = Process(target=run_worker, args=(1,))  # or 1
    worker.start()

    while True:
        time.sleep(1)
        worker.join()