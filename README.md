
# Improving Policy-Conditioned Value Functions
This repository contains the code of my master thesis "Improving Policy-Conditioned Value Functions". The full thesis can be found in `thesis.pdf`. Most of the experiments contained in the Thesis can be simply reproduced with the provided configurations and scripts. In the following the details on how to do so are provided.

## Installation
Mujoco has to be installed separately. Refer to the official repository on instructions how to do so https://github.com/openai/mujoco-py.
Install the requirements in `requirements.txt` with Anaconda via 
```
conda create --name pcvf --file requirements.txt
```
The requirements might have to be adjusted if the use of cuda is desired. 
After installing the requirements execute the following command in the directory of the repository.
```
pip install -e .
```


## Run experiments
The `experiments` folder contains configurations for most experiments of the thesis in a costume format. The script `execute_experiment.py` offers the possibility to run these configurations. For Example 
```
python execute_experiment.py "hc_comp_s_pcac_fp" 50000
```
will run the experiment with the name `hc_comp_s_pcac_fp` for 50000 exploration steps. How many time steps in the environment are executed for one exploration step depends on the specific algorithm. All other parameters are set in the according `config.ini` file. The results of the experiment will be saved in the experiment folder. Rerunning the command will result in a newly initialized run but old runs can also be continued by including the `--run_id` option as follows:
```
python execute_experiment.py "hc_comp_s_pcac_fp" 50000 --run_id 1 --device "cpu"
```
Also `--device` specifies the device to run the experiment on. By default the script will check whether cuda is available and will use it if so. This option may be used to overrule this default.

## Observe trained policies
After executing an experiment the resulting policy can be viewed with the `observe.py` script as following:
```
python observe.py "hc_comp_s_pcac_fp" 1000
```
The `1000` here indicate for how many time steps the demonstration should run. The environment is reset if the episode ends. Here the latest run of the experiment is shown, again with the `--run_id` option a specific run can be shown.
```
python observe.py "hc_comp_s_pcac_fp" 1000 --run_id 1
```
The option `--no-render` disables the rendering and just gives out the mean and standard deviation across all episodes within the demonstration period.

## Experiments
The repository contains the following Experiments:
```
hc_comp_s_pcac_fp
hc_comp_ss_pcac_fp
hc_nstep_s_pcac_fp
hc_psvf
hc_s_pcac_fe_small
hc_s_pcac_fp
hc_s_pcac_fp_small
hc_s_pcac_ne_small
pen_comp_s_pcac_fp
pen_comp_ss_pcac_fp
pen_ma_comp_ss_pcac_fp
pen_ma_nstep_s_pcac_fp
pen_nstep_s_pcac_fp
pen_pavf
pen_pssvf
pen_psvf
pen_s_pcac_fe
pen_s_pcac_fp
pen_s_pcac_ne
```
Shorthands:
<ul>
    <li>hc - HalfCheetah-v2
    <li>pen - Pendulum-v0
    <li>s - state
    <li>ss - start state
    <li>comp - Comparing
    <li>pcac - Policy-conditioned Actor Critic
    <li>ma - Multi Actor
    <li>fp - Fingerprinting
    <li>ne - Neuron embedding
    <li>fe - flat embedding
    <li>nstep - N-step
    <li>pavf - Parameter-Based State-Action Value Function
    <li>pssvf - Parameter-Based Start State Value Function
</ul>
See Thesis for details on individual methods.

## Other
Important bits of the Code are not yet fully documented with Docstrings.

