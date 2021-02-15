import os
import glob
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from os import listdir
from PIL import Image


class AFADDataset(data.Dataset):
    def __init__(self, root, transforms=transforms.Compose([transforms.ToTensor()]), categorical=False):
        super(AFADDataset, self).__init__()
        self.root = root
        self.transforms = transforms
        self.categorical = categorical

        self.age_class_list = None
        self.img_pth_list = None
        self.label_gender_list = None
        self.label_age_list = None
        self.label_dict = {0: 'male', 1: 'female'}

        self.load_annotation_list()

    def __getitem__(self, idx):
        img = Image.open(self.img_pth_list[idx])
        img = self.transforms(img)
        label_age = self.label_age_list[idx]
        label_gender = self.label_gender_list[idx]

        if self.categorical:
            label_age = self.to_categorical(label_age, len(self.age_class_list), self.age_class_list)
            label_gender = self.to_categorical(label_gender, 2)

        label_age = torch.as_tensor(label_age, dtype=torch.int64)
        label_gender = torch.as_tensor(label_gender, dtype=torch.int64)

        ann = {'age': label_age, 'gender': label_gender}

        return img, ann

    def __len__(self):
        return len(self.img_pth_list)

    def load_annotation_list(self):
        print('Loading annotations...')
        img_pth_list = []
        label_gender_list = []  # 0: Male, 1: Female
        label_age_list = []  # Lite: 18-39, Full: 15-75
        age_class_list = []
        age_dir_list = [f for f in list(filter(os.path.isdir, glob.glob(os.path.join(self.root, '*'))))]

        for age_dir in age_dir_list:
            age = int(os.path.split(age_dir)[-1])
            age_class_list.append(age)

            male_dir = os.path.join(age_dir, '111')
            female_dir = os.path.join(age_dir, '112')

            male_pth_list = [os.path.join(male_dir, f) for f in listdir(male_dir)]
            female_pth_list = [os.path.join(female_dir, f) for f in listdir(female_dir)]

            img_pth_list += male_pth_list + female_pth_list
            label_gender_list += [0 for _ in range(len(male_pth_list))] + [1 for _ in range(len(female_pth_list))]
            label_age_list += [age for _ in range(len(male_pth_list) + len(female_pth_list))]

        self.img_pth_list = img_pth_list
        self.label_gender_list = label_gender_list
        self.label_age_list = label_age_list
        self.age_class_list = age_class_list

        print('Annotations loaded!')

    @staticmethod
    def to_categorical(label, num_classes, ref_list=None):
        base = [0 for _ in range(num_classes)]
        if ref_list is not None:
            base[ref_list.index(label)] = 1
        else:
            base[label] = 1

        return base

    @staticmethod
    def custom_collate_fn(batch):
        img = [item[0] for item in batch]
        label = [item[1] for item in batch]

        return img, label
