import torch
import numpy as np
import os
from torch.utils.data import Dataset
import cv2 as cv

class VOC_Dataset(Dataset):
    def __init__(self, desc_feat, img_path, mask_path, img_size=224, transform=None):
        super().__init__()
        self.data = []
        self.transform = transform
        self.desc_dict = desc_feat
        self.img_path = img_path
        self.mask_path = mask_path
        self.img_size = (img_size, img_size)
        self.color_a = np.array([192,224,224])
        self.color_b = np.array([0, 0, 0])
        self.white = np.array([255, 255, 255])
        self.load_data()

    def load_data(self):
        for c in os.listdir(self.img_path):
            cur_class_path = os.path.join(self.img_path, c)
            for img in os.listdir(cur_class_path):
                cur_img_path = os.path.join(cur_class_path, img)
                cur_mask_path = os.path.join(self.mask_path, c, img)
                ## load image
                cur_img = cv.imread(cur_img_path)
                ## load mask
                cur_mask = cv.imread(cur_mask_path)
                ## convert colors
                mask_a = np.all(cur_mask == self.color_a, axis=2)
                cur_mask[mask_a] = self.color_b
                mask_other = np.logical_not(np.logical_or(mask_a, np.all(cur_mask == self.color_b, axis=2)))
                cur_mask[mask_other] = self.white

                cur_img = cv.resize(cur_img / 255, self.img_size).transpose((2, 0, 1))
                cur_mask = cv.resize(cur_mask / 255, self.img_size).transpose((2, 0, 1))
                ## load description
                cur_desc = self.desc_dict[c]['features']
                cur_class = self.desc_dict[c]['class']
                self.data.append({"img": cur_img, "mask": cur_mask[0], "desc": cur_desc, "class": cur_class})

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        cur_data = self.data[idx]
        cur_img = cur_data["img"]
        cur_mask = cur_data["mask"]
        cur_desc = cur_data["desc"]
        cur_class = cur_data["class"]
        if self.transform:
            cur_img = self.transform(cur_img)
            cur_mask = self.transform(cur_mask)

        return cur_img, cur_mask, cur_desc, cur_class