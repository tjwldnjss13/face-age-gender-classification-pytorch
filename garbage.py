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
import cv2 as cv

a = torch.Tensor([1.224, 2.849])
a = a.int()
print(a)






























<<<<<<< HEAD
=======
a = torch.Tensor([1, 2, 3, 4])
b = (a % 2 == 0).sum()
print(b)
>>>>>>> d758cd946f48a0b58e192b1d240a465f3605ad11
