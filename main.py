import os, sys
import glob
import runpy
import time
from multiprocessing import Process
from constants import PYTHON_PATH, NUM_WORKERS
import multiprocessing, logging

if not os.path.exists("./graphs/"):
    os.makedirs("./graphs/")

if not os.path.exists("./logs/"):
    os.makedirs("./logs/")


def run_broker():
    try:
        os.system(PYTHON_PATH + " ./broker.py")
        sys.stdout = open("./out_err/broker_stdout.out", "wb")
        sys.stderr = open("./out_err/broker_stderr.out", "wb")
    except KeyboardInterrupt:
        sys.stdout.flush()
        sys.stderr.flush()


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
        files = glob.glob('./graphs/*')
        for f in files:
            os.remove(f)

        files = glob.glob('./logs/*')
        for f in files:
            os.remove(f)

        files = glob.glob('./out_err/*')
        for f in files:
            os.remove(f)


        broker = Process(target=run_broker, args=())
        broker.start()

        time.sleep(1)

        workers = []
        for worker_id in range(NUM_WORKERS):
            worker = Process(target=run_worker, args=(worker_id,))
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()

        broker.join()