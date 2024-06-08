from xml.dom.minidom import Document
from typing import Optional, List
import xml.etree.ElementTree as ET
import json
from pathlib import Path
import pandas as pd
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import shutil


def dct2coco(categorys, annotations, output_json, h, w, suffix='jpg'):
    images = []
    anno = []
    for stem, v in annotations.items():
        image_id = len(images) + 1
        images.append({
            "file_name": f"{stem}.{suffix}",
            "height": h,
            "width": w,
            "id": image_id
        })
        for x in v:
            x['id'] = 1 + len(anno)
            x['image_id'] = image_id
            anno.append(x)
    with open(output_json, 'w') as fout:
        json.dump({
            "categories": categorys,
            'images': images,
            "annotations": anno,
        }, fout, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    cls_map = {
        1: 'car',
        2: 'truck',
        4: 'tractor',
        5: 'camping car',
        9: 'van',
        10: 'vehicle',
        11: 'pick-up',
        23: 'ship',
        31: 'plane',
        
        'bus': 'bus',
        'cycle': 'cycle'
    }
    categorys = [{
        'id': i,
        'name': v,
        'supercategory': None
    } for i, (k, v) in enumerate(cls_map.items(), 1)]
    cat_id = {x['name']: x['id'] for x in categorys}

    df = pd.read_csv("data/vedai/Annotations512/annotation512.txt", sep=' ', header=None, names=None, dtype={0: str})

    annotations = defaultdict(list)
    for idx, row in df.iterrows():
        rbox = row[4:12].values.reshape(2, 4).T
        x1y1 = rbox.min(0)
        x2y2 = rbox.max(0)
        bbox = x1y1.tolist() + (x2y2-x1y1).tolist()
        if row[12] not in {7, 8}:
            c = cls_map[row[12]]
            annotations[row[0]].append({
                "category_id": cat_id[c],
                "iscrowd": 0,
                "area": np.prod(x2y2 - x1y1),
                "bbox": bbox,
                "segmentation": [bbox]
            })
    dct2coco(categorys, annotations, "data/vedai/annotations.json", 512, 512, 'png')

    imgs = list(Path("data/vedai/Vehicules512").glob("*.png"))
    rgb = Path("data/vedai/Vehicules512/rgb")
    tir = Path("data/vedai/Vehicules512/tir")
    rgb.mkdir(exist_ok=True)
    tir.mkdir(exist_ok=True)
    for x in tqdm(imgs, desc="move and rename imgs"):
        if len(x.stem) == len("00000000_co"):
            stem = x.stem[:-3]
            tp = x.stem[-2:]
            if tp == 'co':
                shutil.move(x, rgb / f"{stem}.png")
            if tp == 'ir':
                shutil.move(x, tir / f"{stem}.png")

    df = pd.read_csv("data/aistudio/data.csv").dropna(how='any')
    annotations = defaultdict(list)
    for idx, row in tqdm(df.iterrows()):
        stem = Path(row.image_name).stem
        x2y2 = np.array([row.x_max, row.y_max])
        x1y1 = np.array([row.x_min, row.y_min])
        bbox = x1y1.tolist() + (x2y2-x1y1).tolist()
        c = row.class_name
        annotations[stem].append({
            "category_id": int(cat_id[c]),
            "iscrowd": 0,
            "area": np.prod(x2y2 - x1y1),
            "bbox": bbox,
            "segmentation": [bbox]
        })
    dct2coco(categorys, annotations, "data/aistudio/annotations.json", 1080, 1920, 'jpg')
