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
    parser.add_argument("-i", "--input", default="data/track1-A/annotations/test.json", type=str)
    parser.add_argument("-r", "--results", default='results.pkl', type=str)
    parser.add_argument("-o", "--output", default='pred.json', type=str)
    parser.add_argument("-t", "--threshold", default=0., type=float)
    return parser.parse_args()


def xyxy2xywh(box):
    # box: [N, 4]
    wh = box[:, [2,3]] -  box[:, [0,1]]
    return torch.cat([box[:, :2], wh], dim=-1)


if __name__ == "__main__":
    args = parse_args()
    results = pkl.load(open(args.results, 'rb'))
    with open(args.input) as fin:
        ijs = json.load(fin)

    submit = []
    for single_img_res in results:
        score = single_img_res['pred_instances']['scores']
        bbox = xyxy2xywh(single_img_res['pred_instances']['bboxes'])
        cate = single_img_res['pred_instances']['labels']
        for i in range(len(score)):
            if score[i] > args.threshold:
                x, y, w, h = bbox[i].tolist()
                # if True:
                if not ((w < 5 and (x<1 or x+w > 639)) or (h<5 and (y<1 or y+h > 511))):
                    submit.append({
                        "image_id": single_img_res['img_id'],
                        "category_id": cate[i].item() + 1,      # 1-based when submit
                        "score": score[i].item(),
                        "bbox": [round(x, 2) for x in bbox[i].tolist()]
                    })
    ## append fake detect to missing image
    already = {x['image_id'] for x in submit}
    fake = 0
    for x in ijs['images']:
        if x['id'] not in already:
            submit.append({
                "image_id": x['id'],
                # "category_id": 1,
                # "score": 1,
                # "bbox": [0,0,0,0]
                "category_id": None,
                "score": None,
                "bbox": None
            })
            already.add(x['id'])
            fake += 1

    # submit = sorted(submit, key=lambda x: x['image_id'])

    with open(args.output, 'w') as fout:
        json.dump(submit, fout, ensure_ascii=False)
    print(f"Mock {fake} image(s).")
    print(f"DONE. Save submit files: `{args.output}`.")
