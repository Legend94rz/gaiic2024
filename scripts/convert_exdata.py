from pathlib import Path
from xml.etree import ElementTree as ET
import json
import numpy as np
from tqdm import tqdm
TRAIN = Path("/home/renzhen/userdata/data/gaiic2024/external/train/")
VAL = Path("/home/renzhen/userdata/data/gaiic2024/external/val/")
CATEGORY = {'car': 1, 'truck': 2, 'bus': 3, 'van': 4, 'freight_car': 5}
OUTPUT = Path("data/external/annotations")


def get_bbox(obj):
    poly = obj.find('polygon')
    if poly is not None:
        g = []
        for i in range(1, 5):
            g.append([int(poly.findtext(f'x{i}')), int(poly.findtext(f'y{i}'))])
        g = np.array(g)
        xmin, ymin = g.min(0).tolist()
        xmax, ymax = g.max(0).tolist()
    else:
        bndbox = obj.find('bndbox')
        xmin = int(bndbox.findtext("xmin"))
        xmax = int(bndbox.findtext("xmax"))
        ymin = int(bndbox.findtext("ymin"))
        ymax = int(bndbox.findtext("ymax"))
    w, h = xmax-xmin, ymax-ymin
    return [xmin, ymin, w, h], w*h


def map_category(cstr):
    cstr = cstr.lower()
    if cstr in {"feright car", "feright_car", "feright"}:
        cstr = "freight_car"
    return CATEGORY[cstr]


def broken(root):
    if any(obj.find('polygon') is None and obj.find('bndbox') is None or obj.findtext('name')=='*' for obj in root.findall('object')):
        return True
    return False


def make_json(xml_dir, filename):
    error = 0
    images = []
    annotations = []
    for i, xml in tqdm(enumerate(sorted(xml_dir.glob("*.xml")), 1)):
        root = ET.parse(xml).getroot()
        if broken(root):
            error += 1
            continue
        images.append({
            "file_name": f"{xml.stem}.jpg",
            "height": int(root.findtext('size/height')),
            "width": int(root.findtext('size/width')),
            "id": i
        })
        for obj in root.findall('object'):
            assert obj.find('polygon') is not None or obj.find('bndbox') is not None, f"ERROR {xml}"
            bbox, area = get_bbox(obj)
            annotations.append({
                "id": len(annotations) + 1,
                "image_id": i,
                "category_id": map_category(obj.findtext("name")),
                "iscrowd": 0,
                "bbox": bbox,
                "segmentation": bbox,
                "area": area,
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
    with open(OUTPUT / filename, 'w') as fout:
        json.dump(obj, fout, indent=2, ensure_ascii=False)
    print(f"errors: {error}")


if __name__ == "__main__":
    make_json(TRAIN / "trainlabelr", "train.json")
    make_json(VAL / "vallabelr", "val.json")