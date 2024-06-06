import os
import cv2
import numpy as np

masks_path = os.path.join(os.getcwd(), "DATA", "zalando-hd-resized", "train_masks")
images_path = os.path.join(os.getcwd(), "DATA", "zalando-hd-resized", "train", "image")

os.makedirs(os.path.join(os.getcwd(), "DATA", "zalando-hd-resized", "dilated_train_masks"), exist_ok=True)

mask_ids = [f for f in os.listdir(masks_path) if os.path.isfile(os.path.join(masks_path, f))]

for mask_name in mask_ids:
    mask = cv2.imread(os.path.join(os.getcwd(), "DATA", "zalando-hd-resized", "train_masks", mask_name))

    image_name = mask_name[:-9] + ".jpg"
    img = cv2.imread(os.path.join(images_path, image_name))

    # Dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=3)
    cv2.imwrite(os.path.join(os.getcwd(), "DATA", "zalando-hd-resized", "dilated_train_masks", mask_name), mask)

    # Flip mask to respect the format of the agnostic mask from viton-hd
    mask = cv2.bitwise_not(mask)

    # Covert mask to greyscale
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Binarize mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Make agnostic mask grey
    agnostic = np.copy(img)
    agnostic[mask == 0] = [128, 128, 128]
    cv2.imwrite(os.path.join(masks_path, image_name), agnostic)
    # cv2.imshow("result", agnostic)
    # cv2.waitKey(0)
