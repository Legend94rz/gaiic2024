from pycocotools.coco import COCO
import argparse
from pathlib import Path
import json
from sklearn.model_selection import KFold


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('test_folder', type=str)
    parser.add_argument('--save_path', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    pass
