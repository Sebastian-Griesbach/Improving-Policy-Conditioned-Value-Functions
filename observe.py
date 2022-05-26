import os
import argparse
import torch
from tqdm import tqdm
import numpy as np

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
            experiment_id = max(list(map(int,existing_experiment_ids)))
        else:
            raise Exception(f"No run for {experiment} found.")

    run_folder = os.path.join(runs_folder, str(experiment_id))
    if not os.path.isdir(run_folder):
        os.mkdir(run_folder)
    return run_folder, config

def main():
    parser = argparse.ArgumentParser(description='Observe a trained Policy.')
    parser.add_argument('experiment', type=str,
                        help='Name of the experiment to be executed.')

    parser.add_argument('time_steps', type=int, help='Number of timesteps to execute.')

    parser.add_argument('--experiment_id', type=int, help='Experiment id to load. Takes latest if none is given.')

    parser.add_argument('--render', dest='render', action='store_true')
    parser.add_argument('--no-render', dest='render', action='store_false')
    parser.set_defaults(render=True)

    parser.add_argument('--device', type=str, help='Cuda device to run model on.', default="cpu")

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
    environment = configuration["environment"]

    checkpoint_folder = os.path.join(run_folder, "checkpoint")

    if os.path.exists(checkpoint_folder):
        trainer.load_checkpoint(checkpoint_folder)
    else:
        raise Exception(f"No checkpoint {checkpoint_folder} found.")

    policy = trainer.get_policy()

    print(f"Loaded policy from Checkpoint {checkpoint_folder}")

    done = True
    cum_rewards = []
    cur_rewards = []
    num_episodes = 0
    for step in tqdm(range(args.time_steps)):
        if(done):
            obs = environment.reset()
            cum_rewards.append(sum(cur_rewards))
            cur_rewards = []
            num_episodes += 1
        action = policy(obs)
        obs, reward, done, _ = environment.step(action)
        if(args.render):
            environment.render()
        cur_rewards.append(reward)

    environment.close()

    reward_mean = np.mean(cum_rewards)
    reward_std = np.std(cum_rewards)

    print(f"Number of episodes {num_episodes}")
    print(f"Mean episode return: {reward_mean} (+/- {reward_std})")
    

if __name__ == "__main__":
    main()