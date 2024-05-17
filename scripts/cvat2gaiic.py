import json
import argparse
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with open(args.file) as fin:
        obj = json.load(fin)
    images = obj["images"]
    for img in images:
        p = Path(img["file_name"])
        assert int(p.stem) == img['id'], img
        img['file_name'] = p.name
    with open(args.file, 'w') as fout:
        json.dump({
            "categories": obj["categories"],
            "images": obj["images"],
            "annotations": obj["annotations"]
        }, fout, ensure_ascii=False, indent=4)