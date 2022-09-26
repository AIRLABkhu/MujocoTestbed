# MujocoTestbed

MujocoTestbed of AIRLAB


## Requirements
Note that: pytorch should folllow CUDA version of your GPU (If not, you may not be able to use pytorch-cuda)
*   [mujoco-py](https://github.com/AIRLABkhu/Manuals/tree/main/Reinforcement%20Learning/Mujoco)
*   [PyTorch](http://pytorch.org/)
* Python >=3.8
* Pytorch >=1.5
* CUDA enabled computing device
* gym==0.14.0

## Build

first intall mujoco-py in your conda env click this link  [mujoco-py](https://github.com/AIRLABkhu/Manuals/tree/main/Reinforcement%20Learning/Mujoco)
<pre><code>
git clone https://github.com/AIRLABkhu/MujocoTestbed.git
pip install -r requirements.txt
</code></pre>

## Algorithm
DDPG,SAC is available for now.

### Default Arguments and Usage
------------
### Usage

```
usage(sac): main.py [-h] [--env-name ENV_NAME] [--policy POLICY] [--eval EVAL]
               [--gamma G] [--tau G] [--lr G] [--alpha G]
               [--automatic_entropy_tuning G] [--seed N] [--batch_size N]
               [--num_steps N] [--hidden_size N] [--updates_per_step N]
               [--start_steps N] [--test_interval][--target_update_interval N]
               [--replay_size N]
```

```
usage(ddpg):main.py [-h] [--env-name ENV_NAME] [--policy POLICY]
               [--gamma G] [--tau G] [--ou_noise] [--noise_scale] [--final_noise_scale G] [--exploration_end N]
               [--lr G] [--alpha G]
               [--automatic_entropy_tuning G] [--seed N] [--batch_size N]
               [--num_steps N] [--hidden_size N] [--updates_per_step N]
               [--start_steps N] [--test_interval][--target_update_interval N]
               [--replay_size N]
```

(Note: There is no need for setting Temperature(`--alpha`) if `--automatic_entropy_tuning` is True.)

#### For SAC

```
python sac/main.py --env-name Humanoid-v2 --alpha 0.05
```

### For DDPG
```
python ddpg/main.py --env-name Huamnoid-v2
```

### Arguments
------------
```
SAC Args

optional arguments:
  -h, --help            show this help message and exit
  --env-name ENV_NAME   Mujoco Gym environment (default: HalfCheetah-v2)
  --policy POLICY       Policy Type: Gaussian | Deterministic (default:
                        Gaussian)
  --eval EVAL           Evaluates a policy a policy every 10 episode (default:
                        True)
  --gamma G             discount factor for reward (default: 0.99)
  --tau G               target smoothing coefficient(τ) (default: 5e-3)
  --lr G                learning rate (default: 3e-4)
  --alpha G             Temperature parameter α determines the relative
                        importance of the entropy term against the reward
                        (default: 0.2)
  --automatic_entropy_tuning G
                        Automaically adjust α (default: False)
  --batch_size N        batch size (default: 256)
  --num_steps N         maximum number of steps (default: 1e6)
  --hidden_size N       hidden size (default: 256)
  --updates_per_step N  model updates per simulator step (default: 1)
  --start_steps N       Steps sampling random actions (default: 1e4)
  --test_interval N evaluation per simulator step (default:5000)
  --target_update_interval N
                        Value target update per no. of updates per step
                        (default: 1)
  --replay_size N       size of replay buffer (default: 1e6)
```

| Environment **(`--env-name`)**| Temperature **(`--alpha`)**|
| ---------------| -------------|
| HalfCheetah-v2| 0.2|
| Hopper-v2| 0.2|
| Walker2d-v2| 0.2|
| Ant-v2| 0.2|
| Humanoid-v2| 0.05|


```
DDPG Args

optional arguments:
  -h, --help            show this help message and exit
  --env-name ENV_NAME   Mujoco Gym environment (default: HalfCheetah-v2)
  --policy POLICY       Policy Type: Gaussian | Deterministic (default:
                        Gaussian)
  --gamma G             discount factor for reward (default: 0.99)
  --tau G               target smoothing coefficient(τ) (default: 1e-3)
  --lr G                learning rate (default: 3e-4)
  --ou_noise            noise for exploration (default:True)
  --noise_scale         initial noise scale (default:0.3)
  --final_noise_scale   final noise scale (default:0.3)
  --exploration_end     number of episodes with noise (default:100)
  --batch_size N        batch size (default: 256)
  --num_steps N         maximum number of steps (default: 1e6)
  --hidden_size N       hidden size (default: 256)
  --updates_per_step N  model updates per simulator step (default: 1)
  --start_steps N       Steps sampling random actions (default: 1e4)
  --test_interval N evaluation per simulator step (default:5000)
  --target_update_interval N
                        Value target update per no. of updates per step
                        (default: 1)
  --replay_size N       size of replay buffer (default: 1e6)
```

### Visualization

First make csv_file
```
python tools.py --root-dir run/{ENV_NAME}/{Algo}
```
and make plot
```
python plotter.py --task {ENV_NAME} --algo {algo1},{algo2},{algo3} (for comparison of algorithms)
```
Example (I run experiments on two algoritms (sac,ddpg) in Ant-v2) then
```
python tools.py --root-dir run/Ant-v2/sac
python tools.py --root-dir run/Ant-v2/ddpg
python plotter.py --task Ant-v2 --algo sac,ddpg
```
