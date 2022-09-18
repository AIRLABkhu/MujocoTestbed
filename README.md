# MujocoTestbed

MujocoTestbed of AIRLAB\
Reference: [Tinashou](https://github.com/thu-ml/tianshou)


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
conda activate mujoco_py
git clone https://github.com/AIRLABkhu/MujocoTestbed.git
pip install -r requirements.txt
</code></pre>

## Algorithm
DDPG,SAC,TD3 is available

## Usage


Run

```bash
$ python mujoco_sac.py --task Ant-v3
```

Logs is saved in `./log/` and can be monitored with tensorboard.

```bash
$ tensorboard --logdir log
```

You can also reproduce the benchmark (e.g. SAC in Ant-v3) with the example script we provide under `examples/mujoco/`:

```bash
$ ./run_experiments.sh Ant-v3 sac
```

This will start 10 experiments with different seeds.

Now that all the experiments are finished, we can convert all tfevent files into csv files and then try plotting the results.

```bash
# generate csv
$ ./tools.py --root-dir ./results/Ant-v3/sac
# generate figures
$ ./plotter.py --root-dir ./results/Ant-v3 --shaded-std --legend-pattern "\\w+"
# generate numerical result (support multiple groups: `--root-dir ./` instead of single dir)
$ ./analysis.py --root-dir ./results --norm
```

## Example benchmark
![offpolicy](https://user-images.githubusercontent.com/75155964/190888083-e77c66f9-68e7-4673-bdb7-c5648da3754c.png)
