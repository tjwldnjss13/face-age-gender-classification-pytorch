import os
import glob
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

from os import listdir
from PIL import Image

from datasets.dataset_sampling import shift_random

######################## AFAD data balance ########################
num_afad_data_dict = {
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
    def __init__(self, root, transform=None, categorical=False, balanced=False):
        super(AFADDataset, self).__init__()
        self.root = root
        self.transform = transform if transform is not None else transforms.ToTensor()
        self.categorical = categorical
        self.balanced = balanced

        self.num_age_classes = 26  # Not range, refer over-40 as 40 as well.
        self.age_class_list = None
        self.img_pth_list = None
        self.label_gender_list = None
        self.label_age_list = None
        self.sampled_img_pth_list = None
        self.sampled_label_gender_list = None
        self.sampled_label_age_list = None
        self.class_num_dict = num_afad_data_dict
        self.label_gender_dict = {0: 'male', 1: 'female'}

        self._load_annotation_list()
        if balanced:
            self._balance_dataset()

        self.age_weight_factor = self.get_age_weight_factor()
        self.gender_weight_factor = self.get_gender_weight_factor()

    def _load_annotation_list(self):
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

    def _balance_dataset(self):
        print('Balancing dataset...')
        sampled_img_pth_list = []
        sampled_label_gender_list = []
        sampled_label_age_list = []

        age_list = list(num_afad_data_dict.keys())
        num_data_list = list(num_afad_data_dict.values())

        max_num_data = max(num_data_list)
        idx_max_num_data = num_data_list.index(max_num_data)
        idx_age_40 = age_list.index(40)
        for i in range(idx_age_40 + 1):
            if i == idx_max_num_data:
                continue

            if i != idx_age_40:
                num_data = num_data_list[i]
            else:
                num_data = sum(num_data_list[i:])
            num_sampled_data = 0
            while num_data < .6 * max_num_data and num_data + num_sampled_data < max_num_data:
                idx_start = sum(num_data_list[:i])
                sampled_img_pth_list += self.img_pth_list[idx_start:idx_start + num_data]
                sampled_label_gender_list += self.label_gender_list[idx_start:idx_start + num_data]
                sampled_label_age_list += self.label_age_list[idx_start:idx_start + num_data]
                num_sampled_data += num_data

        self.sampled_img_pth_list = sampled_img_pth_list
        self.sampled_label_gender_list = sampled_label_gender_list
        self.sampled_label_age_list = sampled_label_age_list

        print('Dataset balanced!!!')

    def __getitem__(self, idx):
        if self.balanced:
            if idx < len(self.img_pth_list):
                img_pth = self.img_pth_list[idx]
                img = Image.open(img_pth)
                label_age = self.label_age_list[idx]
                label_gender = self.label_gender_list[idx]
            else:
                img_pth = self.sampled_img_pth_list[idx - len(self.img_pth_list)]
                img = Image.open(img_pth)
                img = shift_random(img)
                label_age = self.sampled_label_age_list[idx - len(self.img_pth_list)]
                label_gender = self.sampled_label_gender_list[idx - len(self.img_pth_list)]
        else:
            img_pth = self.img_pth_list[idx]
            img = Image.open(img_pth)
            label_age = self.label_age_list[idx]
            label_age = min(label_age, 40)
            label_gender = self.label_gender_list[idx]

        img = self.transform(img)

        # label_age = min(label_age, 40)
        # if label_age <= 16:
        #     label_age = 0
        # elif label_age <= 19:
        #     label_age = 1
        # elif label_age <= 22:
        #     label_age = 2
        # elif label_age <= 24:
        #     label_age = 3
        # elif label_age <= 27:
        #     label_age = 4
        # elif label_age <= 29:
        #     label_age = 5
        # elif label_age <= 32:
        #     label_age = 6
        # elif label_age <= 34:
        #     label_age = 7
        # elif label_age <= 36:
        #     label_age = 8
        # elif label_age <= 38:
        #     label_age = 9
        # else:
        #     label_age = 10

        if self.categorical:
            label_age_cate = self.to_categorical(label_age - 15, self.num_age_classes)
            label_gender_cate = self.to_categorical(label_gender, 2)

            label_age_cate = torch.as_tensor(label_age_cate, dtype=torch.int64)
            label_gender_cate = torch.as_tensor(label_gender_cate, dtype=torch.int64)

        label_age = torch.as_tensor(label_age, dtype=torch.int64)
        label_gender = torch.as_tensor(label_gender, dtype=torch.int64)

        ann = {}
        ann['age'] = label_age
        ann['gender'] = label_gender
        ann['filename'] = img_pth
        if self.categorical:
            ann['age_categorical'] = label_age_cate
            ann['gender_categorical'] = label_gender_cate

        return img, ann

    def __len__(self):
        if self.balanced:
            return len(self.img_pth_list) + len(self.sampled_img_pth_list)
        else:
            return len(self.img_pth_list)

    def get_age_weight_factor(self):
        weight_factor_list = torch.Tensor([0 for _ in range(self.num_age_classes)])
        keys = self.class_num_dict.keys()
        for k in keys:
            weight_factor_list[min(k, 40) - 15] += self.class_num_dict[k]

        sum_num_data = torch.sum(weight_factor_list)
        weight_factor_list = sum_num_data / (26 * weight_factor_list)

        return weight_factor_list

    def get_gender_weight_factor(self):
        weight_factor_list = torch.Tensor([0, 0])
        labels_gender = torch.Tensor(self.label_gender_list)
        for i in range(2):
            weight_factor_list[i] = (labels_gender == i).sum()

        sum_num_data = torch.sum(weight_factor_list)
        weight_factor_list = sum_num_data / (2 * weight_factor_list)

        return weight_factor_list

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






if __name__ == '__main__':
    root = 'C://DeepLearningData/AFAD-Full'
    train_root = os.path.join(root, 'Train')
    test_root = os.path.join(root, 'Test')
    train_dset = AFADDataset(train_root)
    test_dset = AFADDataset(test_root)
    train_wf_gender = train_dset.gender_weight_factor
    print(train_wf_gender)































