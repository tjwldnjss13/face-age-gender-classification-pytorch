import os
import torch
import torch.utils.data as data


class ABGADataset(data.Dataset):
    def __init__(self, root):
        super(super, ABGADataset).__init__()
        self.root = root

        self.img_pth_list = None
        self.label_age_list = None
        self.label_gender_list = None
        self.label_age_ref = ['(0, 2)', '(']