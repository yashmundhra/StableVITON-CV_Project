import albumentations as A
import numpy as np

from cv2 import BORDER_CONSTANT

# Applied to clothing and U-Net's input condition
# image, agn, agn_mask, cloth, cloth_mask, image_densepose, gt_cloth_warped_mask
transform_flip = A.Compose(
    [A.HorizontalFlip(p=0.5)],
    additional_targets={
        'agn': 'image',
        'agn_mask': 'mask',
        'cloth': 'image',
        'cloth_mask': 'mask',
        'image_densepose': 'mask',
        'gt_cloth_warped_mask': 'mask',
    }
)

# Applied to clothing and U-Net's input
# group 1: image, agn, agn_mask, image_densepose, gt_cloth_warped_mask
# group 2: cloth, cloth_mask
transform_shift_scale = A.Compose([
    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, p=0.5, border_mode=BORDER_CONSTANT, value=(0,0,0)),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=0, p=0.5, border_mode=BORDER_CONSTANT, value=(0,0,0))],
    additional_targets={
        # group 1
        'agn': 'image',
        'agn_mask': 'mask',
        'image_densepose': 'mask',
        'gt_cloth_warped_mask': 'mask',

        # group 2 is handled manually in dataset_aug.py
    }
)

# Applied to clothing and x_0
# image, agn, cloth
transform_color = A.Compose([
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.3, p=0.5)],
    additional_targets={
        'cloth': 'image',
        'cloth_mask': 'mask',
    }
)