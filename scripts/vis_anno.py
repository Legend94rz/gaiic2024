import json
import argparse
import cv2
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle as pkl
from torchvision.ops.boxes import box_convert, box_iou


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--anno", default="train.json", type=str, help="train/val/test.json")
    parser.add_argument("-p", "--pred", default=None, help='pred.json if test set.')
    parser.add_argument("-f", "--folder", default="train", type=str, help='image folder')
    parser.add_argument("-s", "--save_path", default="vis", type=str, help="the output folder for vis result")
    parser.add_argument("-t", "--thres", default=0.5, type=float, help="threshold for prediction objects.")
    return parser.parse_args()


def draw_bbox(img, boxes, scores=None):
    palette = {1: (5, 5, 214), 2: (26, 237, 26), 3:(225, 10, 10), 4:(32, 244, 244), 5:(230, 18, 230)}   # RGB
    # palette = [None, (5, 5, 214), (26, 237, 26), (225, 10, 10), (32, 244, 244), (230, 18, 230),  (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (255, 168, 196), (255, 255, 255)]
    for i, box in enumerate(boxes):
        color = palette[box['category_id']]
        pt1 = np.array(box['bbox'][:2])
        pt2 = pt1 + np.array(box['bbox'][2:])
        cv2.rectangle(img, pt1.astype(int).tolist(), pt2.astype(int).tolist(), color[::-1], thickness=3)
        if scores[i] is not None:
            cv2.putText(img, f"{scores[i]:.3f}", (pt1-[0, 4]).astype(int).tolist(), cv2.FONT_HERSHEY_COMPLEX, 0.6, color[::-1], thickness=1)
    return img


def load_anno(i, a):
    with open(a) as fin:
        anno = json.load(fin)

    if i is not None:
        if i.endswith('.json'):
            with open(i) as fin:
                obj = json.load(fin)
            anno['annotations'] = obj
        elif i.endswith('.pkl'):
            with open(i, 'rb') as fin:
                obj = pkl.load(fin)
            ano = []
            for x in obj:
                pred = x['pred_instances']
                bbox = box_convert(pred['bboxes'], 'xyxy', 'xywh').numpy()
                for j in range(len(pred['scores'])):
                    ano.append({
                        "image_id": x['img_id'],
                        "score": pred['scores'][j].item(),
                        "category_id": pred['labels'][j].item() + 1,      # 1-based when submit
                        "bbox": [round(z, 2) for z in bbox[j].tolist()]
                    })
            anno['annotations'] = ano
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
        scores = [(x['score'] if 'score' in x else None) for x in annotations if x['image_id'] == img_id  and ('score' not in x or x['score'] > args.thres)]
        # mat = draw_bbox(cv2.imread(str(f /  "imgs" / img['file_name'])), bboxes, scores)
        # cv2.imwrite(str(s / img['file_name']), mat)
        tir = draw_bbox(cv2.imread(str(f /  "tir" / img['file_name'])), bboxes, scores)
        rgb = draw_bbox(cv2.imread(str(f /  "rgb" / img['file_name'])), bboxes, scores)
        mat = np.concatenate([tir, rgb], axis=0)
        cv2.imwrite(str(s / img['file_name']), mat)
    