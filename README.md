
# Improving Policy-Conditioned Value Functions
## Abstract
Current Actor-Critic Reinforcement Learning algorithms face a fundamental limitation. The critic has to implicitly learn about the current state of the policy embodied by the actor. This in turn leads to a delayed learning reaction of the critic, as it always lacks behind the actor. Policy-Conditioned Value Functions explicitly include a representation of the policy, that has to be evaluated, to the critic. Earlier research has shown that these methods are effective in Reinforcement Learning. In this thesis we build upon the current state of Policy-Conditioned Value Functions by adding several improvements. We explore three different methods for representing policies, which are the flat embedding, the neuron embedding and Network Fingerprinting, and assess their capabilities in different settings. The neuron embedding is our novel approach for policy representations. Furthermore, we establish a base algorithm for Policy-Conditioned Value Functions and explore three algorithmic variants which enable the algorithm to (1) explore multiple policies in one rollout, (2) reframe Reinforcement Learning in continuous space into a binary classification problem and (3) train multiple actors at once with one critic. Our results show that the established policy representation ”Network Fingerprinting” in combination with our algorithmic variants improve the overall performance of Policy-Conditioned Value Functions.

## Code
The code has been developed during my master's thesis "Improving Policy-Conditioned Value Functions". The full thesis can be found in `thesis.pdf`. The aim was to create a framework that would make it easy to implement vastly different Reinforcement Learning algorithms and to compare them. Most of the experiments contained in the thesis can be reproduced with the provided configurations and scripts. In the following the details on how to do so are provided. The Code contains many abstraction layers to make implementation of new algorithms as easy a possible. Also a custom configuration format has been developed that can be used to setup different experimental setting with vastly different class structures. This configuration format has been published as a [python package](https://github.com/Sebastian-Griesbach/Generic-Configuration-Builder). I plan on reorganizing the code further and to build a standalone framework for exotic reinforcement learning algorithm.

## Installation
Mujoco has to be installed separately. Refer to the official repository on instructions how to do so https://github.com/openai/mujoco-py. <p>
Using Anaconda you can create the environment via 
```
conda env create -f environment.yml
```
Alternatively `requirements.txt` contains all exact versions that have been tested. These can be installed with Anaconda via
```
conda create --name pcvf --file requirements.txt
```
The requirements might have to be adjusted if the use of cuda is desired. 
After installing the requirements execute the following command in the directory of the repository.
```
pip install -e .
```


## Run experiments
The `experiments` folder contains configurations for most experiments of the thesis in a custom format. The script `execute_experiment.py` offers the possibility to run these configurations. For example 
```
python execute_experiment.py hc_comp_s_pcac_fp 50000
```
will run the experiment with the name `hc_comp_s_pcac_fp` for 50000 exploration steps. How many time steps in the environment are executed for one exploration step depends on the specific algorithm. All other parameters are set in the according `config.ini` file (here *experiments/hc_comp_s_pcac_fp/config.ini*). At the end of the run a checkpoint is created in the experiment folder with an incremental run ID. This checkpoint may be used to evaluate the policy or continue training. Rerunning the command will result in a newly initialized run. Old runs can also be continued by including the `--run_id` option as follows:
```
python execute_experiment.py hc_comp_s_pcac_fp 50000 --run_id 1 --device "cpu"
```
Also `--device` specifies the device to run the experiment on. By default the script will check whether cuda is available and will use it if so. This option may be used to overrule this default.

## Observe/Evaluate trained policies
After executing an experiment the resulting policy can be viewed with the `observe.py` script as following:
```
python observe.py hc_comp_s_pcac_fp 1000
```
The `1000` here indicate for how many time steps the demonstration should run. The environment is reset if the episode terminates. Here the latest (highest id) run of the experiment is shown, again with the `--run_id` option a specific run can be shown.
```
python observe.py "hc_comp_s_pcac_fp" 1000 --run_id 1 --no-render
```
The option `--no-render` disables the rendering. At the end of the time steps the mean and standard deviation of the return across all episodes is printed.

## Experiments
The following table shows which experiments are contained in this repository and in which figures they (or very similar configurations) have been used. 
|Experiment identifier| Figures |
|---|---|
|hc_comp_s_pcac_fp| 14, 17 |
|hc_comp_ss_pcac_fp| 14 |
|hc_nstep_s_pcac_fp| 13, 17 |
|hc_psvf| 2, 17 |
|hc_s_pcac_fe_small| 8 |
|hc_s_pcac_fp| 13, 14, 17 |
|hc_s_pcac_fp_small| 8 |
|hc_s_pcac_ne_small| 8 |
|pen_comp_s_pcac_fp| 14 |
|pen_comp_ss_pcac_fp| 14, 15, 17 |
|pen_ma_comp_ss_pcac_fp| 15, 16 |
|pen_ma_nstep_s_pcac_fp| 15, 16 |
|pen_nstep_s_pcac_fp| 13, 15, 17 |
|pen_pavf| - |
|pen_pssvf| 2, 17 |
|pen_psvf| 2 |
|pen_s_pcac_fe| 8, 13, 14, 15, 16, 17 |
|pen_s_pcac_fp| 8, 11 |
|pen_s_pcac_ne| 8, 12 |

The names are structured as follows: *environment_algorithm_embedding_other*
<p>Shorthands:
<ul>
    <li>hc - HalfCheetah-v2
    <li>pen - Pendulum-v0
    <li>s - state
    <li>ss - start state
    <li>comp - Comparing
    <li>pcac - Policy-Conditioned Actor Critic
    <li>ma - Multi Actor
    <li>fp - Fingerprinting
    <li>ne - Neuron embedding
    <li>fe - flat embedding
    <li>nstep - N-step
    <li>pavf - Parameter-Based State-Action Value Function
    <li>pssvf - Parameter-Based Start State Value Function
</ul>

See `thesis.pdf` for details on individual methods.

## Documentation
The Reinforcement Learning parts of the code, and some other bits, are documented with DocStrings in the code. If an interface exists, the purpose of all functions are documented within the interface. Specific subclasses are only documented in their general purpose or when a sufficient difference exists.

## Example Policies
A Policy trained with the N-step PCAC algorithm in the Pendulum-v0 environment.
![(Policy learned by N-step PCAC on the Pendulum-v0 environment)](./animations/n-step_pcac_pendulum.gif)

A Policy trained with the Comparing State PCAC algorithm in the HalfCHeetah-v2 environment. Footage is at half speed to make motion more visible.
![(Policy learned by Comparing State PCAC on the HalfCHeetah-v2 environment)](./animations/comp_s_pcac_hc.gif)
