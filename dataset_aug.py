from os.path import join as opj

import cv2 as cv
import numpy as np
from torch.utils.data import Dataset

from aug.augmentation import *

def imread(p, h, w, is_mask=False, in_inverse_mask=False, img=None):
    if img is None:
        img = cv.imread(p)
    if not is_mask:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (w,h))
        img = (img.astype(np.float32) / 127.5) - 1.0  # [-1, 1]
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = cv.resize(img, (w,h))
        img = (img >= 128).astype(np.float32)  # 0 or 1
        img = img[:,:,None]
        if in_inverse_mask:
            img = 1-img
    return img

def image_int_to_float(img, is_mask=False, invert_mask=False):
    if is_mask:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = (img >= 128).astype(np.float32)
        img = img[:,:,None]
        if invert_mask:
            img = 1 - img
    else:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = (img.astype(np.float32) / 127.5) - 1.0
    return img

class VITONHDDataset_aug(Dataset):
    def __init__(
            self, 
            data_root_dir, 
            img_H, 
            img_W, 
            is_paired=True, 
            is_test=False, 
            is_sorted=False,             
            **kwargs
        ):
        self.drd = data_root_dir
        self.img_H = img_H
        self.img_W = img_W
        self.pair_key = "paired" if is_paired else "unpaired"
        self.data_type = "train" if not is_test else "test"
        self.is_test = is_test
       
        assert not (self.data_type == "train" and self.pair_key == "unpaired"), f"train must use paired dataset"
        
        im_names = []
        c_names = []
        with open(opj(self.drd, f"{self.data_type}_pairs.txt"), "r") as f:
            for line in f.readlines():
                im_name, c_name = line.strip().split()
                im_names.append(im_name)
                c_names.append(c_name)
        if is_sorted:
            im_names, c_names = zip(*sorted(zip(im_names, c_names)))
        self.im_names = im_names
        
        self.c_names = dict()
        self.c_names["paired"] = im_names
        self.c_names["unpaired"] = c_names

    def __len__(self):
        return len(self.im_names)
    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]

        agn = cv.imread(opj(self.drd, self.data_type, "agnostic-v3.2", self.im_names[idx]))
        agn_mask = cv.imread(opj(self.drd, self.data_type, "agnostic-mask", self.im_names[idx]))

        cloth = cv.imread(opj(self.drd, self.data_type, "cloth", self.c_names[self.pair_key][idx]))

        image = cv.imread(opj(self.drd, self.data_type, "image", self.im_names[idx]))
        image_densepose = cv.imread(opj(self.drd, self.data_type, "image-densepose", self.im_names[idx]))

        agn, agn_mask, cloth, image, image_densepose =\
            map(lambda img: cv.resize(img, (self.img_W, self.img_H)), [agn, agn_mask, cloth, image, image_densepose])
        
        if not self.is_test: # train
            aug = transform_flip(image=cloth , agn=agn, agn_mask=agn_mask, image_densepose=image_densepose)
            cloth, agn, agn_mask, image_densepose = \
                aug['image'], aug['agn'], aug['agn_mask'], aug['image_densepose']
            
            aug = transform_shift_scale(image=image, cloth=cloth)
            cloth, image = aug['cloth'], aug['image']

            aug = transform_color(image=image, cloth=cloth)
            cloth, image = aug['cloth'], aug['image']
        
        agn, cloth, image, image_densepose = \
            map(image_int_to_float, [agn, cloth, image, image_densepose])
        agn_mask = image_int_to_float(agn_mask, is_mask=True, invert_mask=True)
        
        return dict(
            agn=agn,
            agn_mask=agn_mask,
            cloth=cloth,
            image=image,
            image_densepose=image_densepose,
            txt="",
            img_fn=img_fn,
            cloth_fn=cloth_fn,
        )
