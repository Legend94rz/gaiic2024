from pycocotools.coco import COCO
import argparse
from pathlib import Path
import json
from sklearn.model_selection import KFold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('-n', type=int, default=5, help='#fold')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    kf = KFold(args.n)
    p = Path(args.config)

    with open(p) as fin:
        obj = json.load(fin)
    cats = obj['categories']
    imgs = obj['images']
    ann = obj['annotations']

    for i, (train_idx, val_idx) in enumerate(kf.split(imgs)):
        train_img_ids = {imgs[x]['id'] for x in train_idx}

        train_obj = {
            'categories': cats,
            'images': [imgs[x] for x in train_idx],
            'annotations': [x for x in ann if x['image_id'] in train_img_ids]
        }
        val_obj = {
            'categories': cats,
            'images': [imgs[x] for x in val_idx],
            'annotations': [x for x in ann if x['image_id'] not in train_img_ids]
        }
        with open(p.parent / f"fold_{i}_train.json", 'w') as fout:
            json.dump(train_obj, fout, ensure_ascii=False, indent=4)
        with open(p.parent / f"fold_{i}_val.json", 'w') as fout:
            json.dump(val_obj, fout, ensure_ascii=False, indent=4)
