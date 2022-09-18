import warnings

import gym

from tianshou.env import ShmemVectorEnv, VectorEnvNormObs



def make_mujoco_env(task, seed, training_num, test_num, obs_norm):


    env = gym.make(task)
    train_envs = ShmemVectorEnv(
        [lambda: gym.make(task) for _ in range(training_num)]
    )
    test_envs = ShmemVectorEnv([lambda: gym.make(task) for _ in range(test_num)])
    env.seed(seed)
    train_envs.seed(seed)
    test_envs.seed(seed)

    if obs_norm:
        # obs norm wrapper
        train_envs = VectorEnvNormObs(train_envs)
        test_envs = VectorEnvNormObs(test_envs, update_obs_rms=False)
        test_envs.set_obs_rms(train_envs.get_obs_rms())
    return env, train_envs, test_envs
