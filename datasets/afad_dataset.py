import os
import glob
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from os import listdir
from PIL import Image


######################## AFAD data balance ########################
afad_dict = {
    15: 445 + 806,
    16: 951 + 1482,
    17: 1135 + 1347,
    18: 2553 + 3032,
    19: 6379 + 6089,
    20: 6930 + 6302,
    21: 6960 + 5600,
    22: 7794 + 5778,
    23: 6191 + 4319,
    24: 8146 + 4745,
    25: 7398 + 3863,
    26: 5174 + 2502,
    27: 4469 + 2012,
    28: 3543 + 1627,
    29: 3383 + 1347,
    30: 3415 + 1391,
    31: 3119 + 1337,
    32: 2833 + 1163,
    33: 2597 + 1104,
    34: 2519 + 1337,
    35: 2919 + 1327,
    36: 2943 + 1291,
    37: 2653 + 1245,
    38: 2713 + 1085,
    39: 2585 + 1206,
    40: 1008 + 327,
    41: 89 + 25,
    42: 114 + 32,
    43: 100 + 18,
    44: 70 + 24,
    45: 51 + 14,
    46: 51 + 8,
    47: 37 + 12,
    48: 39 + 9,
    49: 21 + 9,
    50: 16 + 12,
    51: 17 + 13,
    52: 12 + 11,
    53: 28 + 12,
    54: 13 + 14,
    55: 6 + 6,
    56: 7 + 9,
    57: 6 + 5,
    58: 7 + 5,
    59: 4 + 3,
    60: 1 + 5,
    61: 5 + 3,
    62: 12 + 13,
    63: 5 + 3,
    64: 7 + 1,
    65: 8 + 1,
    66: 15 + 1,
    67: 6 + 4,
    68: 6 + 1,
    69: 13 + 2,
    70: 4 + 3,
    72: 4 + 31
}


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

        # ann = {'age': label_age, 'gender': label_gender}
        label = torch.cat([label_gender, label_age], dim=0)

        return img, label

    def __len__(self):
        return len(self.img_pth_list)

    def load_annotation_list(self):
        print('Loading annotations...')
        img_pth_list = []
        label_gender_list = []  # 0: Male, 1: Female
        label_age_list = []  # Lite: 18-39, Full: 15-70,72
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
