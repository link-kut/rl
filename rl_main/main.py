import sys, os
import time
from multiprocessing import Process

idx = os.getcwd().index("{0}rl".format(os.sep))
PROJECT_HOME = os.getcwd()[:idx+1] + "rl{0}".format(os.sep)
sys.path.append(PROJECT_HOME)

from rl_main.main_constants import *

import rl_main.utils as utils


if __name__ == "__main__":
    utils.make_output_folders()
    utils.print_configuration()
    utils.ask_file_removal()

    stderr = sys.stderr
    sys.stderr = sys.stdout

    try:
        chief = Process(target=utils.run_chief, args=())
        chief.start()

        time.sleep(1.5)

        workers = []
        for worker_id in range(NUM_WORKERS):
            worker = Process(target=utils.run_worker, args=(worker_id,))
            workers.append(worker)
            worker.start()

        for worker in workers:
            worker.join()

        chief.join()
    except KeyboardInterrupt as error:
        print("=== {0:>8} is aborted by keyboard interrupt".format('Main'))
    finally:
        sys.stderr = stderr
