import os
import argparse
import torch

from tc_configurator.configuration_builder import GenericConfigurationBuilder

def _find_run_folder_and_config(experiment, experiment_id):
    experiment_folder = os.path.abspath(f"./experiments/{experiment}")
    config = os.path.join(experiment_folder, "config.ini")

    runs_folder = os.path.join(experiment_folder, "runs")
    if(not os.path.isdir(runs_folder)):
        os.mkdir(runs_folder)
    
    if(experiment_id == None):
        existing_experiment_ids = os.listdir(runs_folder)
        if len(existing_experiment_ids) > 0:
            experiment_id = max(list(map(int,existing_experiment_ids))) + 1
        else:
            experiment_id = 1

    run_folder = os.path.join(runs_folder, str(experiment_id))
    if not os.path.isdir(run_folder):
        os.mkdir(run_folder)
    return run_folder, config

def main():
    parser = argparse.ArgumentParser(description='Executing or continuing an experiment with a specific configuration.')
    parser.add_argument('experiment', type=str,
                        help='Name of the experiment to be executed.')

    parser.add_argument('train_steps', type=int,
                        help='Number of train steps to execute the config on.')

    parser.add_argument('--experiment_id', type=int, help='Experiment id to be continued. Creates new run if None.')

    parser.add_argument('--device', type=str, help="Cuda device to execute experiment on.", default= "cuda" if torch.cuda.is_available() else "cpu")

    args = parser.parse_args()

    run_folder, config = _find_run_folder_and_config(args.experiment, args.experiment_id)

    config_inputs_dict = {
    "configuration_path": config,
    "log_dir": run_folder,
    "device": args.device
    }

    configurator = GenericConfigurationBuilder()
    configuration = configurator.build(**config_inputs_dict)

    trainer = configuration["trainer"]
    #environment = configuration["environment"]
    #log_handler = configuration["log_handler"]

    checkpoint_folder = os.path.join(run_folder, "checkpoint")

    if os.path.exists(checkpoint_folder):
        trainer.load_checkpoint(checkpoint_folder)
        print(f"Continuing from checkpoint {checkpoint_folder}")

    print(f"Starting Training in {run_folder}")
    trainer.train(args.train_steps)
    
    if not os.path.isdir(checkpoint_folder):
        os.mkdir(checkpoint_folder)

    print(f"Saving Checkpoint {checkpoint_folder}")
    trainer.save_checkpoint(checkpoint_folder)

if __name__ == "__main__":
    main()