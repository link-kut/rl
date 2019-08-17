import sys
from multiprocessing import Process

import rl_main.utils as utils
from rl_main.chief_workers.chief import env


if __name__ == "__main__":
    utils.make_output_folders()
    utils.print_configuration(env)
    utils.ask_file_removal()

    stderr = sys.stderr
    sys.stderr = sys.stdout

    try:
        chief = Process(target=utils.run_chief, args=())
        chief.start()
        chief.join()
    except KeyboardInterrupt as error:
        print("=== {0:>8} is aborted by keyboard interrupt".format('Main-Chief'))
    finally:
        sys.stderr = stderr
