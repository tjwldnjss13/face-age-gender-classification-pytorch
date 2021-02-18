import os
import glob
import torch

from torch.utils.data import random_split

from datasets.afad_dataset import AFADDataset, afad_dict
from os import listdir

# root = 'D://DeepLearningData/AFAD-Full/'
# dset = AFADDataset(root=root)
#
# n_data = len(dset)
# n_train_data = int(n_data * .7)
# train_val_ratio = [n_train_data, n_data - n_train_data]
# train_dset, val_dset = random_split(dset, train_val_ratio)
# print(val_dset.transform)

ages, num_data = list(afad_dict.keys()), list(afad_dict.values())

import matplotlib.pyplot as plt
fig = plt.figure(0)
plt.bar([i for i in range(len(ages))], num_data)
plt.xlabel('Ages')
plt.ylabel('Num data')
plt.xticks([i for i in range(len(ages))], ages)
plt.show()