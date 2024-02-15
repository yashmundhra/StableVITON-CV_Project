import albumentations as A
import numpy as np

from cv2 import BORDER_CONSTANT

# Applied to clothing and U-Net's input condition
transform_flip = A.Compose(
    [A.HorizontalFlip(p=0.5)],
    additional_targets={
        'agn': 'image',
        'agn_mask': 'mask',
        'image_densepose': 'mask'
    }
)

# Applied to clothing and U-Net's input
transform_shift_scale = A.Compose([
    A.ShiftScaleRotate(shift_limit=0, scale_limit=0.2, rotate_limit=0, p=0.5, border_mode=BORDER_CONSTANT, value=(0,0,0)),
    A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0, rotate_limit=0, p=0.5, border_mode=BORDER_CONSTANT, value=(0,0,0))],
)

# Applied to clothing and x_0
transform_color = A.Compose([
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.3, p=0.5)],
    additional_targets={
        'cloth': 'image'
    }
)