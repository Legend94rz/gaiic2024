from xml.dom.minidom import Document
from typing import Optional, List
import xml.etree.ElementTree as ET
import json
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    with open('/home/renzhen/userdata/repo/gaiic2024/data/track1-A/annotations/val.json') as fin:
        src = json.load(fin)
    labelme_json = Path("/home/renzhen/userdata/repo/gaiic2024/data/track1-A/val/json")
    output_file = Path("/home/renzhen/userdata/repo/gaiic2024/data/track1-A/annotations/val_updated.json")

    output = {}
    new_anno = []
    label2id = {x['name']: x['id'] for x in src['categories']}
    for img in tqdm(src['images']):
        stem = Path(img['file_name']).stem
        with open(labelme_json / f'{stem}.json') as fin:
            updated = json.load(fin)
        for x in updated['shapes']:
            x1, y1 = x['points'][0]
            x2, y2 = x['points'][1]
            new_anno.append({
                "id": len(new_anno) + 1,
                "image_id": img['id'],
                "category_id": label2id[x['label']],
                "bbox": [
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1
                ],
                "area": (y2 - y1) * (x2-x1),
                "segmentation": [],
                "iscrowd": 0
            })
    src['annotations'] = new_anno
    with open(output_file, 'w') as fout:
        json.dump(src, fout, ensure_ascii=False, indent=4)
