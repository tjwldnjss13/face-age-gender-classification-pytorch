import os
import glob
import torch.utils.data as data
import torchvision.transforms as transforms

from os import listdir
from PIL import Image


class AFADDataset(data.Dataset):
    def __init__(self, root, transforms=transforms.Compose([transforms.ToTensor()])):
        super(AFADDataset, self).__init__()
        self.root = root
        self.transforms = transforms
        self.img_pth_list, self.label_list = self.get_annotation_list()
        self.label_dict = {0: 'male', 1: 'female'}

    def __getitem__(self, idx):
        img = Image.open(self.img_pth_list[idx])
        img = self.transforms(img)
        label = self.label_list[idx]

        return img, label

    def get_annotation_list(self):
        print('Loading annotations...')
        img_pth_list = []
        label_list = []  # 0: Male, 1: Female
        age_dir_list = [f for f in list(filter(os.path.isdir, glob.glob(os.path.join(self.root, '*'))))]
        for age_dir in age_dir_list:
            male_dir = os.path.join(age_dir, '111')
            female_dir = os.path.join(age_dir, '112')

            male_pth_list = [os.path.join(male_dir, f) for f in listdir(male_dir)]
            female_pth_list = [os.path.join(female_dir, f) for f in listdir(female_dir)]

            img_pth_list += male_pth_list + female_pth_list
            label_list += [0 for _ in range(len(male_pth_list))] + [1 for _ in range(len(female_pth_list))]

        print('Annotations loaded!')

        return img_pth_list, label_list

    @staticmethod
    def custom_collate_fn(batch):
        img = [item[0] for item in batch]
        label = [item[1] for item in batch]

        return img, label
