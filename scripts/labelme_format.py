from xml.dom.minidom import Document
from typing import Optional, List
import xml.etree.ElementTree as ET
import json
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from pathlib import Path
from tqdm import tqdm


if __name__ == "__main__":
    coco = COCO('/home/renzhen/userdata/repo/gaiic2024/data/track1-A/annotations/train.json')
    save_dir = Path("/home/renzhen/userdata/repo/gaiic2024/data/track1-A/train/json")
    save_dir.mkdir(parents=True, exist_ok=True)

    LABEL_NAMES = ('[PLACEHOLDER]', 'car', 'truck', 'bus', 'van', 'freight_car')
    for img_id in tqdm(coco.getImgIds()):
        ann = coco.loadAnns(coco.getAnnIds(img_id))
        img = coco.loadImgs(img_id)[0]
        f = Path(img['file_name'])
        obj = {
            "version": "4.0.0",
            "flags": {},
            "shapes": [
                {
                    "label": LABEL_NAMES[x['category_id']],
                    "points": [
                        [x['bbox'][0], x['bbox'][1]],
                        [x['bbox'][0] + x["bbox"][2], x["bbox"][1] + x["bbox"][3]]
                    ],
                    "group_id": None,
                    "shape_type": "rectangle",
                    "flags": {}
                } for x in ann
            ],
            "imagePath": f.name,
            "imageData": None,
            "imageHeight": img['height'],
            "imageWidth": img['width']
        }


        with open(save_dir / f"{f.stem}.json", 'w') as fout:
            json.dump(obj, fout, ensure_ascii=False, indent=2)
