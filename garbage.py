import os
import glob
import torch

from datasets.afad_dataset import AFADDataset
from os import listdir

# root = 'D://DeepLearningData/AFAD-Full/'
# dset = AFADDataset(root=root)

a = torch.Tensor([[1.2, 2.4, 0.5], [3.2, 1.5, 532.3]])
a_argmax = a.argmax(dim=1)
print(a_argmax)
