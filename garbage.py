import os
import glob

from dataset.afad_dataset import AFADDataset
from os import listdir

root = 'D://DeepLearningData/AFAD-Lite/'
dset = AFADDataset(root)
img, label = dset[0]
print(img)
print(label)