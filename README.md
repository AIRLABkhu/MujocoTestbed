# MujocoTestbed
MujocoTestbed


## Requirements
Note that: conda env and pytorch should folllow CUDA version of your GPU (If not, you may not be able to use pytorch-cuda)
*   [mujoco-py](https://github.com/AIRLABkhu/Manuals/tree/main/Reinforcement%20Learning/Mujoco)
*   [PyTorch](http://pytorch.org/)
* Python >=3.8
* Pytorch >=1.5
* CUDA enabled computing device
* gym==0.14.0

## Build
<pre><code>
conda activate mujoco_py
git clone https://github.com/AIRLABkhu/MujocoTestbed.git
pip install -r requirements.txt
</code></pre>

## Usage
<pre><code>

</code></pre>

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

<img src="./benchmark/Ant-v3/offpolicy.png" width="500" height="450">

Other graphs can be found under `examples/mujuco/benchmark/`

For pretrained agents, detailed graphs (single agent, single game) and log details, please refer to [https://cloud.tsinghua.edu.cn/d/f45fcfc5016043bc8fbc/](https://cloud.tsinghua.edu.cn/d/f45fcfc5016043bc8fbc/).

## Offpolicy algorithms

#### Notes

1. In offpolicy algorithms (DDPG, TD3, SAC), the shared hyperparameters are almost the same, and unless otherwise stated, hyperparameters are consistent with those used for benchmark in SpinningUp's implementations (e.g. we use batchsize 256 in DDPG/TD3/SAC while SpinningUp use 100. Minor difference also lies with `start-timesteps`, data loop method `step_per_collect`, method to deal with/bootstrap truncated steps because of timelimit and unfinished/collecting episodes (contribute to performance improvement), etc.).
2. By comparison to both classic literature and open source implementations (e.g., SpinningUp)<sup>[[1]](#footnote1)</sup><sup>[[2]](#footnote2)</sup>, Tianshou's implementations of DDPG, TD3, and SAC are roughly at-parity with or better than the best reported results for these algorithms, so you can definitely use Tianshou's benchmark for research purposes.
3. We didn't compare offpolicy algorithms to OpenAI baselines [benchmark](https://github.com/openai/baselines/blob/master/benchmarks_mujoco1M.htm), because for now it seems that they haven't provided benchmark for offpolicy algorithms, but in [SpinningUp docs](https://spinningup.openai.com/en/latest/spinningup/bench.html) they stated that "SpinningUp implementations of DDPG, TD3, and SAC are roughly at-parity with the best-reported results for these algorithms", so we think lack of comparisons with OpenAI baselines is okay.
