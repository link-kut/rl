import time
from multiprocessing import Process
import rl_main.utils as utils

from rl_main.chief_workers.chief import env
from rl_main.conf.constants_mine import NUM_WORKERS


if __name__ == "__main__":
    utils.make_output_folders()
    utils.print_configuration(env)
    utils.ask_file_removal()

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
