import os, sys
import glob
import time
from multiprocessing import Process
from conf.constants_general import PYTHON_PATH, NUM_WORKERS

if not os.path.exists("./graphs/"):
    os.makedirs("./graphs/")

if not os.path.exists("./logs/"):
    os.makedirs("./logs/")

if not os.path.exists("./out_err/"):
    os.makedirs("./out_err/")

def run_broker():
    try:
        os.system(PYTHON_PATH + " ./broker.py")
        sys.stdout = open("./out_err/broker_stdout.out", "wb")
        sys.stderr = open("./out_err/broker_stderr.out", "wb")
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

    while True:
        time.sleep(1)
        broker.join()
