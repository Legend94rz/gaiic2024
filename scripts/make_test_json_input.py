from pycocotools.coco import COCO
import argparse
from pathlib import Path
import json
from sklearn.model_selection import KFold
CATEGORY = {'car': 1, 'truck': 2, 'bus': 3, 'van': 4, 'freight_car': 5}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_folder', type=str)
    parser.add_argument('--save_path', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    fd = Path(args.test_folder) / "rgb"
    output = Path(args.save_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    for i, fimg in enumerate(sorted(fd.glob("*.jpg")), 1):
        images.append({
            "file_name": fimg.name,
            "height": 512,
            "width": 640,
            "id": i
        })
        annotations.append({
            "id": len(annotations) + 1,
            "image_id": i,
            "category_id": 1,
            "iscrowd": 0,
            "bbox": [0, 0, 1, 1],
            "segmentation": [],
            "area": 1,
        })
    
    obj = {
        "categories": [
            {
                "id": v,
                "name": k,
                "supercategory": "None"
            } for k, v in CATEGORY.items()
        ],
        "images": images,
        "annotations": annotations
    }
    with open(output, 'w') as fout:
        json.dump(obj, fout, indent=2, ensure_ascii=False)
