import json
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import numpy as np
import torch
import pickle as pkl
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=str, help="result.pkl")
    parser.add_argument("output", type=str, help="submit.json")
    parser.add_argument("-t", "--threashold", default=0.5, type=float)
    return parser.parse_args()


def xyxy2xywh(box):
    # box: [N, 4]
    wh = box[:, [2,3]] -  box[:, [0,1]]
    cxy = (box[:, [0, 1]] + box[:, [2, 3]]) / 2
    return torch.cat([cxy, wh], dim=-1)


if __name__ == "__main__":
    args = parse_args()
    results = pkl.load(open(args.results, 'rb'))

    submit = []
    for single_img_res in results:
        score = single_img_res['pred_instances']['scores']
        bbox = xyxy2xywh(single_img_res['pred_instances']['bboxes'])
        cate = single_img_res['pred_instances']['labels']
        for i in range(len(score)):
            if score[i] > args.threshold:
                submit.append({
                    "image_id": single_img_res['img_id'],
                    "category_id": cate[i].item(),
                    "score": score[i].item(),
                    "bbox": [round(x, 2) for x in bbox[i].tolist()]
                })
    with open(args.output, 'w') as fout:
        json.dump(submit, fout, ensure_ascii=False, indent=4)
    print(f"DONE. Save submit files: `{args.output}`")
