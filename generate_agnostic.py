import os
import cv2
import numpy as np

ids = ["06"]  # , "08", "13", "17", "34", "35", "55", "57"
for id in ids:
    image_name = f"000{id}_00.jpg"
    img = cv2.imread(os.path.join(os.getcwd(), f"DATA/zalando-hd-resized/test/image/{image_name}"))

    mask_name = f"000{id}_00_mask.jpg"
    mask = cv2.imread(os.path.join(os.getcwd(), f"masks/{mask_name}"))

    # Dilate mask
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (5, 5))
    mask = cv2.dilate(mask, kernel, iterations=3)

    # Flip mask to respect the format of the agnostic mask from viton-hd
    mask = cv2.bitwise_not(mask)
    # cv2.imwrite(os.path.join(os.getcwd(), f"DATA/zalando-hd-resized/masks/{mask_name}"), flip_mask)

    # Covert mask to greyscale
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    # Binarize mask
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    mask = cv2.resize(mask, (img.shape[1], img.shape[0]))

    # Make agnostic mask grey
    agnostic = np.copy(img)
    agnostic[mask == 0] = [128, 128, 128]
    cv2.imwrite(os.path.join(os.getcwd(), f"masks/{image_name}"), agnostic)
    # cv2.imshow("result", agnostic)
    # cv2.waitKey(0)

    # Agnostic mask is black
    # agnostic = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imwrite(os.path.join(os.getcwd(), f"DATA/zalando-hd-resized/masks/{image_name}"), agnostic)
    # cv2.imshow("result", agnostic)
    # cv2.waitKey(0)
