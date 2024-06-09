import json
from os import path as osp
import os
import sys
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm

def get_img_agnostic(img, parse, pose_data):
    parse_array = np.array(parse)
    parse_head = ((parse_array == 4).astype(np.float32) +
                    (parse_array == 13).astype(np.float32))
    parse_lower = ((parse_array == 9).astype(np.float32) +
                    (parse_array == 12).astype(np.float32) +
                    (parse_array == 16).astype(np.float32) +
                    (parse_array == 17).astype(np.float32) +
                    (parse_array == 18).astype(np.float32) +
                    (parse_array == 19).astype(np.float32))

    
    agnostic = img.copy()
    agnostic_draw = ImageDraw.Draw(agnostic)

    mask = Image.new('L', img.size)
    mask_draw = ImageDraw.Draw(mask)

    length_a = np.linalg.norm(pose_data[5] - pose_data[2])
    length_b = np.linalg.norm(pose_data[12] - pose_data[9])
    point = (pose_data[9] + pose_data[12]) / 2
    pose_data[9] = point + (pose_data[9] - point) / length_b * length_a
    pose_data[12] = point + (pose_data[12] - point) / length_b * length_a
    r = int(length_a / 16) + 1
    
    # mask arms
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'gray', width=r*10)
    mask_draw.line([tuple(pose_data[i]) for i in [2, 5]], 'white', width=r*10)
    for i in [2, 5]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        mask_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'white', 'white')
    for i in [3, 4, 6, 7]:
        if (pose_data[i - 1, 0] == 0.0 and pose_data[i - 1, 1] == 0.0) or (pose_data[i, 0] == 0.0 and pose_data[i, 1] == 0.0):
            continue
        agnostic_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'gray', width=r*10)
        mask_draw.line([tuple(pose_data[j]) for j in [i - 1, i]], 'white', width=r*10)

        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'gray', 'gray')
        mask_draw.ellipse((pointx-r*5, pointy-r*5, pointx+r*5, pointy+r*5), 'white', 'white')

    # mask torso
    for i in [9, 12]:
        pointx, pointy = pose_data[i]
        agnostic_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'gray', 'gray')
        mask_draw.ellipse((pointx-r*3, pointy-r*6, pointx+r*3, pointy+r*6), 'white', 'white')
    agnostic_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'gray', width=r*6)
    agnostic_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'gray', width=r*12)
    agnostic_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'gray', 'gray')
    mask_draw.line([tuple(pose_data[i]) for i in [2, 9]], 'white', width=r*6)
    mask_draw.line([tuple(pose_data[i]) for i in [5, 12]], 'white', width=r*6)
    mask_draw.line([tuple(pose_data[i]) for i in [9, 12]], 'white', width=r*12)
    mask_draw.polygon([tuple(pose_data[i]) for i in [2, 5, 12, 9]], 'white', 'white')

    # mask neck
    pointx, pointy = pose_data[1]
    agnostic_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'gray', 'gray')
    mask_draw.rectangle((pointx-r*7, pointy-r*7, pointx+r*7, pointy+r*7), 'white', 'white')
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    agnostic.paste(img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))
    
    black_img = Image.new('L', img.size)
    mask.paste(black_img, None, Image.fromarray(np.uint8(parse_head * 255), 'L'))
    mask.paste(black_img, None, Image.fromarray(np.uint8(parse_lower * 255), 'L'))

    return agnostic, mask

if __name__ =="__main__":
    data_path, output_path, mask_path = sys.argv[1:]

    # data_path = './data'
    # output_path = './data/agnostic-v3.2'
    # mask_path = './data/agnostic-mask'
    
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(mask_path, exist_ok=True)
    
    for im_name in tqdm(os.listdir(osp.join(data_path, 'image'))):
        
        # load pose image
        pose_name = osp.splitext(im_name)[0] + '_keypoints.json'

        try:
            with open(osp.join(data_path, 'openpose_json', pose_name), 'r') as f:
                pose_label = json.load(f)
                pose_data = pose_label['people'][0]['pose_keypoints_2d']
                pose_data = np.array(pose_data)
                pose_data = pose_data.reshape((-1, 3))[:, :2]
        except IndexError:
            print(pose_name)
            continue

        # load parsing image
        im = Image.open(osp.join(data_path, 'image', im_name))
        label_name = osp.splitext(im_name)[0] + '.jpg'
        im_label = Image.open(osp.join(data_path, 'image-parse-v3', label_name))

        agnostic, mask = get_img_agnostic(im, im_label, pose_data)
        agnostic.save(osp.join(output_path, im_name))
        mask.save(osp.join(mask_path, im_name))