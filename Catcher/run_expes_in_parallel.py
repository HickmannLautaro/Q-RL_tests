import argparse
import json
import os
import subprocess
import multiprocessing

import time

import numpy as np
from tqdm import trange


def get_parsed(text=None):
    parser = argparse.ArgumentParser(description="Config for parallel experiment runner")
    parser.add_argument('--parallel', type=int, help="number of parallel processes", default=2)
    parser.add_argument('--start_new', action="store_true", help="Boolean to see if list to resume exists or not (True means always a new list is started)", default=False)

    arguments = vars(parser.parse_args())

    return arguments


def save_list_to_file(list_of_experiments):
    try:
        os.makedirs("utils/parallel_script_tmp/")
    except OSError:
        pass
    list_of_experiments_file = open("utils/parallel_script_tmp/list_of_experiments_file.json", "w")
    json.dump(list_of_experiments, list_of_experiments_file)
    list_of_experiments_file.close()


def convert_dict_to_str(expe):
    project = "Catcher-Simplified"
    name = f"Run-{expe['run']}"
    arg_mod = expe["name"]
    message_name = f"{project}/{arg_mod}/{name}"
    log_file = os.path.join("Saves", message_name)
    # args = [
    #     ["--layers 32 32 --name Classic_32_32"],
    #     ["--layers 64 --name Classic_64"],
    #     ["--Quantum --n_layer 15 --name Quntum_15"],
    #     ["--Quantum --n_layer 10 --name Quntum_10"],
    #     ["--Quantum --n_layer 5  --name Quntum_5"],
    # ]

    if expe["type"] == "Quantum":
        args = "--Quantum "
    else:
        args = ""

    del expe["type"]

    args += "--" + " --".join([" ".join([key, str(val)]) for key, val in expe.items()])
    command = f"PYTHONUNBUFFERED=1 python train_catcher.py "
    try:
        os.makedirs(log_file)
    except OSError:
        pass

    return command + args + f" 1> {log_file}/out.log 2> {log_file}/err.log"


def run_expe(command):
    print(f"Now running: {command}")
    p = subprocess.Popen(command, shell=True)
    p.wait()

    list_of_experiments_file = open("utils/parallel_script_tmp/list_of_experiments_file.json", "r")
    list_of_experiments = json.loads(list_of_experiments_file.read())
    list_of_experiments.remove(command)
    save_list_to_file(list_of_experiments)

    return command


def main():
    try:
        list_of_experiments_file = open("utils/parallel_script_tmp/list_of_experiments_file.json", "r")
        list_of_experiments = json.loads(list_of_experiments_file.read())
    except:
        list_of_experiments = []
    args = get_parsed()
    # configs = [
    #     [{"type": "Quantum", "n_layer": "15", "name": "Quantum_15"}, [1, 6]],
    #     [{"type": "Quantum", "n_layer": "10", "name": "Quantum_10"}, [1, 6]],
    #     [{"type": "Quantum", "n_layer": "5", "name": "Quantum_5"}, [1, 6]],
    #     [{"type": "Classic", "layers": "32 32", "name": "Classic_32_32"}, [1, 6]],
    #     [{"type": "Classic", "layers": "64", "name": "Classic_64"}, [1, 6]],
    # ]

    configs = [
        # [{"type": "Quantum", "n_layer": "10", "name": "Quantum_10_analytical"}, [1, 2]],
        [{"type": "Quantum", "n_layer": "10", "name": "Quantum_10_sampled_5000", "Shots": "5000"}, [2, 6]],
        # [{"type": "Quantum", "n_layer": "10", "name": "Quantum_10_sampled_1000", "Shots": "1000"}, [1, 6]],
        # [{"type": "Quantum", "n_layer": "10", "name": "Quantum_10_sampled_100", "Shots": "100"}, [1, 6]],
    ]

    processes = args["parallel"]

    myorder = []
    if len(list_of_experiments) > 0 and not args["start_new"]:
        print("Resuming last experiment list")
    else:
        print("Starting new experiment list")
        list_of_experiments = []

        count = 0
        count_old= 0
        indices = []
        for c in configs:
            count += c[1][1] - c[1][0]
            indices.append(np.arange(count_old, count).tolist())
            count_old= count

        while np.sum(indices) != 0.0:
            for c in indices:
                if len(c)>0:
                    myorder += [c.pop(0)]

        for config in configs:
            for run in range(config[1][0], config[1][1]):
                expe = config[0].copy()
                expe["run"] = run

                list_of_experiments.append(convert_dict_to_str(expe))

    if len(myorder) > 0:
        list_of_experiments = [list_of_experiments[i] for i in myorder]

    print(len(list_of_experiments))
    print(*list_of_experiments, sep="\n")

    # runing_processes = []
    #
    #
    # for i in trange(0, len(list_of_experiments), processes):
    #
    #     for command in list_of_experiments[i:i + processes]:
    #         print(f"Now running: {command}")
    #         process = subprocess.Popen(command, shell=True)
    #         runing_processes.append(process)
    #         time.sleep(5)  # Deface processes slightly to avoid resources conflict
    #     # Collect statuses
    #     output = [p.wait() for p in runing_processes]
    #     save_list_to_file(list_of_experiments[i + processes:])
    save_list_to_file(list_of_experiments)

    pool = multiprocessing.Pool(processes=processes)
    print("-" * 100, "\nStarting experiments \n")

    '''run experiments in parallel depending on Pool(processes) and when all (n parallel processes) return removes the items from the json list.
    json file is saved and reopened so that in case the program crashes the current stat of finished experiments is automatically saved.
    If all experiments were successfully completed the list should be empty.
    '''

    # TODO See how to remove element when run is finished without waiting for parallel runs to finish.
    for result in pool.imap(run_expe, list_of_experiments):
        # list_of_experiments_file = open("utils/parallel_script_tmp/list_of_experiments_file.json", "r")
        # list_of_experiments = json.loads(list_of_experiments_file.read())
        # list_of_experiments.remove(result)
        # save_list_to_file(list_of_experiments)
        pass

    print("All done")


if __name__ == "__main__":
    main()
