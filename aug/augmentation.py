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

def transform_shift_scale_pad(**data):
    image = data['image']

    hborder = image[[0, -1], :].reshape(-1, 3)
    vborder = image[1:-1, [0, -1]].reshape(-1, 3)
    borders = np.concatenate([hborder, vborder])

    # unique, counts = np.unique(borders, axis=0, return_counts=True)
    # majority_color = unique[counts.argsort()[0]]

    # pad_color = majority_color
    pad_color = borders.mean(axis=0)
    
    image = transform_shift_scale(image=image)['image']
    image[image.sum(axis=2)==0] = pad_color
    
    return image


# Applied to clothing and x_0
transform_color = A.Compose([
    A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=0.3, p=0.5)],
    additional_targets={
        'cloth': 'image'
    }
)