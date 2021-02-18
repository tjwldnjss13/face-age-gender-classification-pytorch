import os
import sys
import random
import numpy as np
import torch
import torch.utils.data as data
import xml.etree.ElementTree as Et

from PIL import Image
from xml.etree.ElementTree import Element, ElementTree


class VOCDataset(data.Dataset):
    def __init__(self, root, img_size, segmentation=False, shuffle_seg=False, transforms=None, is_categorical=True):
        self.root = root
        self.img_size = img_size
        self.segmentation = segmentation
        self.transforms = transforms
        self.is_categorical = is_categorical
        self.num_classes = 20
        self.class_dict_with_bg = {'background': 0, 'aeroplane': 1, 'bicycle': 2, 'bird': 3, 'boat': 4, 'bottle': 5, 'bus': 6, 'car': 7, 'cat': 8,
                                   'chair': 9, 'cow': 10, 'diningtable': 11, 'dog': 12, 'horse': 13, 'motorbike': 14, 'person': 15,
                                   'pottedplant': 16, 'sheep': 17, 'sofa': 18, 'train': 19, 'tvmonitor': 20, 'ambigious': 255}
        self.class_dict = {'aeroplane': 0, 'bicycle': 1, 'bird': 2, 'boat': 3, 'bottle': 4, 'bus': 5, 'car': 6,
                           'cat': 7, 'chair': 8, 'cow': 9, 'diningtable': 10, 'dog': 11, 'horse': 12, 'motorbike': 13,
                           'person': 14, 'pottedplant': 15, 'sheep': 16, 'sofa': 17, 'train': 18, 'tvmonitor': 19}

        if segmentation:
            self.filenames = self.make_seg_image_list()
            if shuffle_seg:
                random.shuffle(self.filenames)
        if not segmentation:
            self.annotation = self.make_ann_list()

    def __getitem__(self, idx):
        if self.segmentation:
            img_dir = os.path.join(self.root, 'JPEGImages')
            img_pth = os.path.join(img_dir, self.filenames[idx])
            img = Image.open(img_pth)

            gt_dir = os.path.join(self.root, 'SegmentationClass')
            gt_pth = os.path.join(gt_dir, self.filenames[idx])
            gt = Image.open(gt_pth)

            if self.transforms is not None:
                img = self.transforms(img)
                gt = self.transforms(gt)
            else:
                img = torch.as_tensor(img, dtype=torch.float64)
                gt = torch.as_tensor(gt, dtype=torch.float64)

            return img, gt
        else:
            img_dir = os.path.join(self.root, 'JPEGImages')
            ann = self.annotation[idx]
            img_fn = ann.find('filename').text
            img_path = os.path.join(img_dir, img_fn)
            img = Image.open(img_path).convert('RGB')
            img_np = np.array(img)

            if self.transforms is not None:
                img = self.transforms(img)
            else:
                img = torch.as_tensor(img, dtype=torch.float64)

            h_img, w_img = img_np.shape[0], img_np.shape[1]
            h_in, w_in = self.img_size[0], self.img_size[1]
            ratio_h, ratio_w = h_in / h_img, w_in / w_img

            objs = ann.findall('object')
            labels = []
            bboxes = []
            for obj in objs:
                name = obj.find('name').text
                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text) / w_img
                ymin = float(bbox.find('ymin').text) / h_img
                xmax = float(bbox.find('xmax').text) / w_img
                ymax = float(bbox.find('ymax').text) / h_img

                label = self.class_dict[name]
                # if self.is_categorical:
                #     class_ = self.to_categorical(class_, self.num_classes)
                #     class_ = torch.as_tensor(class_)

                bbox = torch.as_tensor([ymin, xmin, ymax, xmax])

                labels.append(label)
                bboxes.append(bbox)

            # label = torch.as_tensor(label, dtype=torch.int64)
            # bbox = torch.as_tensor(bbox, dtype=torch.float32)

            if self.is_categorical:
                labels = self.to_categorical(labels, self.num_classes)

            labels = torch.as_tensor(labels, dtype=torch.int64)

            ann = {'label': labels, 'bbox': bboxes, 'filename': img_fn}

            return img, ann

    def __len__(self):
        if self.segmentation:
            return len(self.filenames)
        else:
            return len(self.annotation)

    def make_ann_list(self):
        print('Loading annotations...')
        ann_dir = os.path.join(self.root, 'Annotations')
        ann_fns = os.listdir(ann_dir)

        anns = []
        for i, ann_fn in enumerate(ann_fns):
            # 디버깅할 때 시간 단축 용
            # if i >= 100:
            #     break
            ann_path = os.path.join(ann_dir, ann_fn)
            ann = open(ann_path, 'r')
            tree = Et.parse(ann)
            root_ = tree.getroot()
            anns.append(root_)
            ann.close()

        print('Annotations loaded!')

        return anns

    def make_seg_image_list(self):
        fn_txt_dir = os.path.join(self.root, 'ImageSets', 'Segmentation')

        if self.is_validation:
            fn_txt_pth = os.path.join(fn_txt_dir, 'train.txt')
        else:
            fn_txt_pth = os.path.join(fn_txt_dir, 'val.txt')

        with open(fn_txt_pth) as file:
            fns = file.readlines()

        for i in range(len(fns)):
            fns[i] = str.strip(fns[i])

        return fns

    @staticmethod
    def to_categorical(label, num_classes):
        label_result = [0 for _ in range(num_classes)]
        if isinstance(label, list):
            for l in label:
                label_result[l] = 1
        else:
            label_result[label] = 1

        # label_ = [0 for _ in range(n_class)]
        # label_[label] = 1

        return label_result


def custom_collate_fn(batch):
    images = [item[0] for item in batch]
    anns = [item[1] for item in batch]

    return [images, anns]

