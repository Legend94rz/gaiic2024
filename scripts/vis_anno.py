import json
import argparse
import cv2
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--anno", default="train.json", type=str, help="train/val/test.json")
    parser.add_argument("-p", "--pred", default=None, help='pred.json if test set.')
    parser.add_argument("-f", "--folder", default="train", type=str, help='image folder')
    parser.add_argument("-s", "--save_path", default="vis", type=str, help="the output folder for vis result")
    parser.add_argument("-t", "--thres", default=0.5, type=int, help="threshold for prediction objects.")
    return parser.parse_args()


def draw_bbox(img, boxes):
    palette = {1: (5, 5, 214), 2: (26, 237, 26), 3:(225, 10, 10), 4:(32, 244, 244), 5:(230, 18, 230)}   # RGB
    for box in boxes:
        color = palette[box['category_id']]
        pt1 = np.array(box['bbox'][:2])
        pt2 = pt1 + np.array(box['bbox'][2:])
        cv2.rectangle(img, pt1.astype(int).tolist(), pt2.astype(int).tolist(), color[::-1], thickness=3)
    return img


def load_anno(i, a):
    with open(a) as fin:
        anno = json.load(fin)

    if i is not None:
        with open(i) as fin:
            obj = json.load(fin)
        anno['annotations'] = obj
    images = {
        x['id']: x
        for x in anno['images']
    }
    annotations = anno['annotations']
    return images, annotations


if __name__ == "__main__":
    args = parse_args()
    images, annotations = load_anno(args.pred, args.anno)
    f = Path(args.folder)
    s = Path(args.save_path)
    s.mkdir(parents=True, exist_ok=True)

    for img_id, img in tqdm(images.items()):
        bboxes = [x for x in annotations if x['image_id'] == img_id and ('score' not in x or x['score'] > args.thres)]
        tir = draw_bbox(cv2.imread(str(f /  "tir" / img['file_name'])), bboxes)
        rgb = draw_bbox(cv2.imread(str(f /  "rgb" / img['file_name'])), bboxes)
        mat = np.concatenate([tir, rgb], axis=0)
        cv2.imwrite(str(s / img['file_name']), mat)
    