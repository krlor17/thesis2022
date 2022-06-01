import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

kappa_simple = np.loadtxt("../runs/alpha/logdata/valKappa_simple_lr075.csv", skiprows = 1, delimiter=",")[:,2]
kappa_alpha = np.loadtxt("../runs/alpha/logdata/valKappa_alpha_lr075.csv", skiprows = 1, delimiter=",")[:,2]
kappa_tied = np.loadtxt("../runs/tied_rnn/logdata/valKappa_tiedRNN_frozen.csv", skiprows = 1, delimiter=",")[:,2]

kappa_cnn = np.loadtxt("../runs/cnn/logdata/valKappa_cnn.csv", skiprows = 1, delimiter=",")[:,2]
kappa_const =  np.loadtxt("../runs/placebo/logdata/valKappa_constant_frozen.csv", skiprows = 1, delimiter=",")[:,2]
kappa_rand =  np.loadtxt("../runs/placebo/logdata/valKappa_random_lr050.csv", skiprows = 1, delimiter=",")[:,2]

plt.plot(kappa_simple)
plt.plot(kappa_alpha)
plt.plot(kappa_tied)
# plt.plot(kappa_cnn)

#%%

np.correlate(kappa_simple - np.mean(kappa_simple), kappa_tied- np.mean(kappa_tied))
np.correlate(kappa_simple, kappa_alpha)

np.correlate(kappa_simple, kappa_cnn)

#%%
ref = kappa_simple

# plt.figure()
# plt.plot(avg)
# plt.plot(kappa_simple)

# plt.figure()
# plt.plot(avg)
# plt.plot(kappa_tied)

# plt.figure()
# plt.plot(avg)
# plt.plot(kappa_alpha)

# plt.figure()
# plt.plot(avg)
# plt.plot(kappa_cnn)

# mse_simple = np.sqrt(np.sum(np.power(kappa_simple - avg, 2)))
mse_alpha = np.sqrt(np.sum(np.power(kappa_alpha - ref, 2)))
mse_tied = np.sqrt(np.sum(np.power(kappa_tied - ref, 2)))
mse_cnn = np.sqrt(np.sum(np.power(kappa_cnn - ref, 2)))
mse_const = np.sqrt(np.sum(np.power(kappa_const - ref, 2)))
mse_rand = np.sqrt(np.sum(np.power(kappa_rand - ref, 2)))

mse = np.array([ mse_alpha, mse_tied, mse_cnn, mse_const, mse_rand])
order = np.argsort(mse)
mse = mse[order]
names = np.array([ "Train. alpha", "Tied RNN", "CNN", "Constant", "Random"])[order]
# sns.boxplot(mse, color="white")
# sns.swarmplot(mse, hue=["Simple alpha", "Train. alpha", "Tied RNN", "CNN"])
plt.figure(figsize=(5,5))
df = pd.DataFrame([mse], columns = names)
sns.swarmplot(data=df, color="black", size=8, orient="h" )
plt.hlines(np.arange(len(mse)), 0, df.values, color="k")
plt.rc('ytick', labelsize=18)
plt.rc('xtick', labelsize=12)
plt.xlabel("MSE", fontsize=18)

#%%

ref = kappa_cnn
mse_simple = np.sqrt(np.sum(np.power(kappa_simple - ref, 2)))
mse_alpha = np.sqrt(np.sum(np.power(kappa_alpha - ref, 2)))
mse_tied = np.sqrt(np.sum(np.power(kappa_tied - ref, 2)))
mse_const = np.sqrt(np.sum(np.power(kappa_const - ref, 2)))
mse_rand = np.sqrt(np.sum(np.power(kappa_rand - ref, 2)))

mse = np.array([ mse_simple, mse_alpha, mse_tied,  mse_const, mse_rand])
order = np.argsort(mse)
mse = mse[order]
names = np.array(["Simple", "Train. alpha", "Tied RNN", "Constant", "Random"])[order]
# sns.boxplot(mse, color="white")
# sns.swarmplot(mse, hue=["Simple alpha", "Train. alpha", "Tied RNN", "CNN"])
plt.figure(figsize=(5,5))
df = pd.DataFrame([mse], columns = names)
sns.swarmplot(data=df, color="black", size=8, orient="h" )
plt.hlines(np.arange(len(mse)), 0, df.values, color="k")
plt.rc('ytick', labelsize=18)
plt.rc('xtick', labelsize=12)
plt.xlabel("MSE", fontsize=18)

#%% unfrozen

# kappa_simple_unf = np.loadtxt("../runs/alpha/logdata/valKappa_simple_lr075.csv", skiprows = 1, delimiter=",")[:,2]
kappa_alpha_unf = np.loadtxt("../runs/alpha/logdata/valKappa_unfrozen_lr075.csv", skiprows = 1, delimiter=",")[:,2]
kappa_tied_unf = np.loadtxt("../runs/tied_rnn/logdata/valKappa_tiedRNN_unfrozen.csv", skiprows = 1, delimiter=",")[:,2]

kappa_cnn_unf = np.loadtxt("../runs/cnn/logdata/valKappa_cnn-unfrozen.csv", skiprows = 1, delimiter=",")[:,2]
kappa_const_unf =  np.loadtxt("../runs/placebo/logdata/valKappa_const_unfrozen_lr050.csv", skiprows = 1, delimiter=",")[:,2]
kappa_rand_unf =  np.loadtxt("../runs/placebo/logdata/valKappa_random_unfrozen.csv", skiprows = 1, delimiter=",")[:,2]

#%%
ref = kappa_alpha_unf

# mse_alpha = np.sqrt(np.sum(np.power(kappa_alpha_unf - ref, 2)))
mse_tied = np.sqrt(np.sum(np.power(kappa_tied_unf - ref, 2)))
mse_const = np.sqrt(np.sum(np.power(kappa_const_unf - ref, 2)))
mse_rand = np.sqrt(np.sum(np.power(kappa_rand_unf - ref, 2)))
mse_cnn = np.sqrt(np.sum(np.power(kappa_cnn_unf - ref, 2)))
mse = np.array([  mse_cnn, mse_tied,  mse_const, mse_rand])
order = np.argsort(mse)
mse = mse[order]
names = np.array(["CNN",  "Tied RNN", "Constant", "Random"])[order]
# sns.boxplot(mse, color="white")
# sns.swarmplot(mse, hue=["Simple alpha", "Train. alpha", "Tied RNN", "CNN"])
plt.figure(figsize=(5,5))
df = pd.DataFrame([mse], columns = names)
sns.swarmplot(data=df, color="black", size=8, orient="h" )
plt.hlines(np.arange(len(mse)), 0, df.values, color="k")
plt.rc('ytick', labelsize=18)
plt.rc('xtick', labelsize=12)
plt.xlabel("MSE", fontsize=18)

#%%

ref = kappa_cnn_unf

mse_alpha = np.sqrt(np.sum(np.power(kappa_alpha_unf - ref, 2)))
mse_tied = np.sqrt(np.sum(np.power(kappa_tied_unf - ref, 2)))
mse_const = np.sqrt(np.sum(np.power(kappa_const_unf - ref, 2)))
mse_rand = np.sqrt(np.sum(np.power(kappa_rand_unf - ref, 2)))
# mse_cnn = np.sqrt(np.sum(np.power(kappa_cnn_unf - ref, 2)))
mse = np.array([  mse_alpha, mse_tied,  mse_const, mse_rand])
order = np.argsort(mse)
mse = mse[order]
names = np.array(["Train. alpha",  "Tied RNN", "Constant", "Random"])[order]
# sns.boxplot(mse, color="white")
# sns.swarmplot(mse, hue=["Simple alpha", "Train. alpha", "Tied RNN", "CNN"])
plt.figure(figsize=(5,5))
df = pd.DataFrame([mse], columns = names)
sns.swarmplot(data=df, color="black", size=8, orient="h" )
plt.hlines(np.arange(len(mse)), 0, df.values, color="k")
plt.rc('ytick', labelsize=18)
plt.rc('xtick', labelsize=12)
plt.xlabel("MSE", fontsize=18)

#%% visual comparisons
plt.rc('ytick', labelsize=12)
plt.rc('xtick', labelsize=12)
## frozen
plt.figure(figsize=(5,5))
plt.plot(kappa_simple, label="Simple alpha")
plt.plot(kappa_alpha, label="Train. alpha")
plt.plot(kappa_tied, label="Tied RNN")
# plt.plot(kappa_cnn_unf)
plt.legend()
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Validation Kappa", fontsize=16)

plt.figure(figsize=(5,5))
plt.plot(1/3*kappa_alpha + 1/3*kappa_tied + 1/3*kappa_simple, label="Mean")
plt.plot(kappa_cnn, label="CNN")
plt.legend()
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Validation Kappa", fontsize=16)

## unfrozen
plt.figure(figsize=(5,5))
plt.plot(kappa_alpha_unf, label="Train. alpha")
plt.plot(kappa_tied_unf, label="Tied RNN")
# plt.plot(kappa_cnn_unf)
plt.legend()
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Validation Kappa", fontsize=16)

plt.figure(figsize=(5,5))
plt.plot(0.5*kappa_alpha_unf + 0.5*kappa_tied_unf, label="Mean")
plt.plot(kappa_cnn_unf, label="CNN")
plt.legend()
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Validation Kappa", fontsize=16)