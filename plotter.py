

#!/usr/bin/env python3

import os
import re
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import seaborn as sns
import argparse
import string

parser = argparse.ArgumentParser(description='Plot Generator')
parser.add_argument('--task')
parser.add_argument('--algo',type=str)

args=parser.parse_args()
algo=args.algo.split(',')
csv_algo_lst = list()
csv_env_step = list()
csv_reward = list()
def smooth_data(data, window_size):
    return data.rolling(window=window_size, min_periods=1).mean()
    
for a in algo:
    df1 = pd.read_csv('./run/{}/{}/test_reward_seeds.csv'.format(args.task,a))
    df1 = df1.values.tolist()
    for lst in df1:
        for i in lst[3:]:
            csv_algo_lst.append(a)
            csv_env_step.append(lst[0])
            csv_reward.append(i)

my_df=pd.DataFrame({"Model":csv_algo_lst,"timesteps":csv_env_step,"Accumulated Reward":csv_reward})
my_df['Accumulated Reward'] = smooth_data(my_df['Accumulated Reward'], 1)

sns.lineplot(x="timesteps",y="Accumulated Reward",hue="Model",data=my_df)
plt.show()

sns.lineplot(x="timesteps",y="Accumulated Reward",hue="Model",data=my_df)
plt.show()
