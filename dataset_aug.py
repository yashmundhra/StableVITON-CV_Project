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
    
    def get_image_path(self, dir, idx):
        return opj(self.drd, self.data_type, dir, self.im_names[idx])
    
    def __getitem__(self, idx):
        img_fn = self.im_names[idx]
        cloth_fn = self.c_names[self.pair_key][idx]

        item = dict(
            agn = cv.imread(self.get_image_path("agnostic-v3.2", idx)),
            agn_mask = cv.imread(self.get_image_path("agnostic-mask", idx).replace('.jpg', '_mask.png')),

            cloth = cv.imread(self.get_image_path("cloth", idx)),
            cloth_mask = cv.imread(self.get_image_path('cloth-mask', idx)),

            image = cv.imread(self.get_image_path('image', idx)),
            image_densepose = cv.imread(self.get_image_path('image-densepose', idx)),

            gt_cloth_warped_mask = cv.imread(self.get_image_path('gt_cloth_warped_mask', idx).replace('.jpg', '.png')),
        )

        # resize all images/masks
        for k in item.keys():
            item[k] = cv.resize(item[k], (self.img_W, self.img_H))
        
        if not self.is_test: # train
            # apply transform_flip
            target_keys = ['image', 'agn', 'agn_mask', 'cloth', 'cloth_mask', 'image_densepose']
            item.update(transform_flip(**{k:item[k] for k in target_keys}))

            # apply transform_shift_scale
            # group 1
            target_keys = ['image', 'agn', 'agn_mask', 'image_densepose', 'gt_cloth_warped_mask']
            item.update(transform_shift_scale(**{k:item[k] for k in target_keys}))
            # group 2
            aug = transform_shift_scale(image=item['cloth'], mask=item['cloth_mask'])
            item['cloth'], item['cloth_mask'] = aug['image'], aug['mask']

            # apply transform_color
            target_keys = ['image', 'agn', 'cloth']
            item.update(transform_color(**{k:item[k] for k in target_keys}))
        
        # invert agn_mask
        item['agn_mask'] = 255 - item['agn_mask']
        # why?
        # agn = agn * agn_mask[:,:,None].astype(np.float32)/255.0 + 128 * (1 - agn_mask[:,:,None].astype(np.float32)/255.0)

        # normalization
        for k in item.keys():
            item[k] = image_int_to_float(item[k], is_mask=('mask' in k))
        
        return dict(
            txt="",
            img_fn=img_fn,
            cloth_fn=cloth_fn,
            **item
        )
