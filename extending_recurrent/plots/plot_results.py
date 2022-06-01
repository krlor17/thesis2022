import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

BASE = "/media/kafkan/storage1/SeqSleepNetExtension/extending_recurrent/cross_validate"

epoch_const = np.load(f"{BASE}/placebo/rundata/epoch-extension-constant/epoch-extension-constant/kappas.npy")
epoch_rand = np.load(f"{BASE}/placebo/rundata/epoch-extension-random/epoch-extension-random/kappas.npy")
epoch_cnn = np.load(f"{BASE}/cnn/rundata/epoch-extension/epoch-extension/kappas.npy")
epoch_alpha = np.load(f"{BASE}/alpha/rundata/epoch-extension-alpha/epoch-extension-alpha/kappas.npy")

epoch = pd.DataFrame(data=[epoch_const, epoch_rand, epoch_cnn, epoch_alpha]).T
epoch = epoch.rename(columns = {0:"Constant", 1:"Random", 2:"CNN", 3:"Alpha"})

sequence_const = np.load(f"{BASE}/placebo/rundata/sequence-extension-constant/sequence-extension-constant/kappas.npy")
sequence_rand = np.load(f"{BASE}/placebo/rundata/sequence-extension-random/sequence-extension-random/kappas.npy")
sequence_cnn = np.load(f"{BASE}/cnn/rundata/sequence-extension/sequence-extension/kappas.npy")
sequence_alpha = np.load(f"{BASE}/alpha/rundata/sequence-extension-alpha/sequence-extension-alpha/kappas.npy")

sequence = pd.DataFrame(data=[sequence_const, sequence_rand, sequence_cnn, sequence_alpha]).T
sequence = sequence.rename(columns = {0:"Constant", 1:"Random", 2:"CNN", 3:"Alpha"})

#%% plot options

SMALL_SIZE = 12
MEDIUM_SIZE = 16
BIGGER_SIZE = 18

#%% epoch block extension
plt.figure()

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# sns.boxplot(data=df, palette="Set2", )
sns.boxplot(data=epoch, palette=sns.palettes.light_palette((0.5,0.5,0.5), n_colors=1), )

sns.swarmplot(data=epoch, color=".25")
plt.ylabel("Cohen's Kappa")

max_kappa = max(epoch.max())
min_kappa = max(epoch.min())
diff = max_kappa - min_kappa
means = epoch.mean()
ax = plt.gca()
for x, mean in enumerate(means):
    ax.text(x,max_kappa + 0.1*diff,f"{mean:.3f}", ha="center")
#%% sequence block extension
plt.figure()

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
# sns.boxplot(data=df, palette="Set2", )
sns.boxplot(data=sequence, palette=sns.palettes.light_palette((0.5,0.5,0.5), n_colors=1), )

sns.swarmplot(data=sequence, color=".25")
plt.ylabel("Cohen's Kappa")

max_kappa = max(sequence.max())
min_kappa = max(sequence.min())
diff = max_kappa - min_kappa
means = sequence.mean()
ax = plt.gca()
for x, mean in enumerate(means):
    ax.text(x,max_kappa + 0.1*diff,f"{mean:.3f}", ha="center")
    

#%%



plt.rc('ytick', labelsize=MEDIUM_SIZE)  
plt.rc('xtick', labelsize=MEDIUM_SIZE)  

conf_seq_alpha = np.load(f"{BASE}/alpha/rundata/sequence-extension-alpha/sequence-extension-alpha/confusion_matrices.npy")
conf_seq_const = np.load(f"{BASE}/placebo/rundata/sequence-extension-constant/sequence-extension-constant/confusion_matrices.npy")

conf_alpha = conf_seq_alpha[6].astype(int)
conf_const = conf_seq_const[6].astype(int)

tick = ["W","R","N1","N2","N3"]

plt.figure(figsize=(5,5))
sns.heatmap(conf_alpha, cbar=False, annot=True,
            fmt="d",xticklabels=tick, yticklabels=tick, cmap="Blues" )

plt.figure(figsize=(5,5))
sns.heatmap(conf_const, cbar=False, annot=True,
            fmt="d",xticklabels=tick, yticklabels=tick, cmap="Blues" )

plt.figure(figsize=(5,5))
sns.heatmap(conf_alpha-conf_const, cbar=False, annot=True,
            fmt="d",xticklabels=tick, yticklabels=tick,
            cmap=sns.diverging_palette(240,10, n=7), )

#%% metrics
def conf_matrix_metrics(conf, verbose=False):
    ticks = ["W","R","N1","N2","N3"]

    metrics = []
    for i, label in enumerate(ticks):
        TP = conf[i,i]
        FP = sum(conf[:,i]) - TP
        FN = sum(conf[i,:]) - TP
        prec = TP/(TP+FP)
        recall = TP/(TP+FN)
        prop = sum(conf[i,:])/np.sum(conf)
        if verbose:
            print(label,f"\t precision:{prec:.4f}\t recall:{recall:.4f}\t")
        metrics.append([label, prec, recall, prop])
    table = pd.DataFrame(metrics, columns=["label","precision", "recall", "proportion"])
    return table

metc =conf_matrix_metrics(conf_const)
meta = conf_matrix_metrics(conf_alpha)
print(metc.to_latex())
print(meta.to_latex())

diff = meta.values[:,1:3] - metc.values[:,1:3]
print(diff)

#%%
conf_seq_rand = np.load(f"{BASE}/placebo/rundata/sequence-extension-random/sequence-extension-random/confusion_matrices.npy")
conf_seq_cnn = np.load(f"{BASE}/cnn/rundata/sequence-extension/sequence-extension/confusion_matrices.npy")

conf_rand = conf_seq_rand[6].astype(int)
conf_cnn = conf_seq_cnn[6].astype(int)

metcnn =conf_matrix_metrics(conf_cnn)
metrand = conf_matrix_metrics(conf_rand)
print(metcnn.to_latex())
print(metrand.to_latex())
