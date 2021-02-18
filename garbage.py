import os
import glob
import torch

from torch.utils.data import random_split

from datasets.afad_dataset import AFADDataset
from os import listdir

# root = 'D://DeepLearningData/AFAD-Full/'
# dset = AFADDataset(root=root)
#
# n_data = len(dset)
# n_train_data = int(n_data * .7)
# train_val_ratio = [n_train_data, n_data - n_train_data]
# train_dset, val_dset = random_split(dset, train_val_ratio)
# print(val_dset.transform)

import numpy as np

a = [1, 2, 3, 4, 5, 6, 7]
print(a[3:])