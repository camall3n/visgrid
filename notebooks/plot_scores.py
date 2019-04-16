import glob
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from gridworlds.utils import load_experiment

def load_experiment(tag):
    logfiles = sorted(glob.glob(os.path.join('logs',tag+'*','scores-*.txt')))
    seeds = [int(f.split('-')[-1].split('.')[0]) for f in logfiles]
    logs = [open(f,'r').read().splitlines() for f in logfiles]
    def read_log(log):
        results = [json.loads(item) for item in log]
        data = pd.DataFrame(results)
        return data
    results = [read_log(log) for log in logs]
    data = pd.concat(results, join='outer', keys=seeds, names=['seed']).sort_values(by='seed', kind='mergesort').reset_index(level=0)
    return data

labels = ['tag']
experiments = ['test-nofac-6x6']
data = pd.concat([load_experiment(e) for e in experiments], join='outer', keys=experiments, names=labels).reset_index(level=[0])

g = sns.relplot(x='episode', y='reward', kind='line', units='trial', estimator=None, data=data, legend=False, height=4)
plt.subplots_adjust(top=0.9)
g.fig.suptitle('Reward vs. Time')
plt.show()