import os

def main():
    experiment_name = ...
    experiment_folder = os.path.abspath("./experiments/{experiment_name}")
    config = os.path.join(experiment_folder, "config.ini")

    runs_folder = os.path.join(experiment_folder, "runs")
    if(not os.path.isdir(runs_folder)):
        os.mkdir(runs_folder)
    
    run_folder = ...

if __name__ == "__main__":
    main()