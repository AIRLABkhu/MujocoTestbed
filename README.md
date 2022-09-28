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

#### For SAC

```
python SAC/main.py --env-name Humanoid-v2 --alpha 0.05
```

### For DDPG
```
python DDPG/main.py --env-name Huamnoid-v2
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
