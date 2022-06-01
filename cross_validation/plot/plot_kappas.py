"""
    Plot results of cross validation
"""
import os
os.chdir("/media/kafkan/storage1/SeqSleepNetExtension/cross_validation/plot")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

kappas_const = np.load("../placebo/rundata/constant-unfrozen/kappas.npy")
kappas_rand = np.load("../placebo/rundata/random-unfrozen/kappas.npy")
kappas_cnn = np.load("../cnn/rundata/CNN-fresh/kappas.npy")
kappas_tied = np.load("../tiedRNN/rundata/tiedRNN-unfrozen/kappas.npy")
kappas_alpha = np.load("../alpha/rundata/alpha/kappas.npy")


df = pd.DataFrame(data=[kappas_const, kappas_rand, kappas_cnn, kappas_tied, kappas_alpha]).T
df = df.rename(columns={0:"Constant",1:"Random", 2:"CNN",3:"Tied RNN", 4:"Alpha"})
SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18
plt.figure(figsize=(8,5))
plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# sns.boxplot(data=df, palette="Set2", )
sns.boxplot(data=df, palette=sns.palettes.light_palette((0.5,0.5,0.5), n_colors=1), )

sns.swarmplot(data=df, color=".25")
plt.ylabel("Cohen's Kappa")

max_kappa = max(df.max())
min_kappa = max(df.min())
diff = max_kappa - min_kappa
means = df.mean()
ax = plt.gca()
for x, mean in enumerate(means):
    ax.text(x,max_kappa + 0.1*diff,f"{mean:.3f}", ha="center")


