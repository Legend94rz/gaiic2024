import json
import argparse
import cv2
from pathlib import Path
import numpy as np
from itertools import chain
ratio = 3


def split(img, ratio):
    img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio)
    h = np.split(img, ratio, axis=0)
    for i in range(len(h)):
        h[i] = np.split(h[i], ratio, axis=1)
    return list(chain.from_iterable(h))


if __name__ == "__main__":
    rgb_list = [
        f'data/track1-A/test/rgb/{i:05}.jpg'
        for i in range(792, 880)
    ]

    for f in rgb_list:
        stem = Path(f).stem
        rgb = cv2.imread(f)
        for i, mat in enumerate(split(rgb, ratio)):
            cv2.imwrite(f'data/track1-A/_small_obj/rgb/{stem}_{i}.jpg', mat)

        tir = cv2.imread(f.replace('rgb', 'tir'))
        for i, mat in enumerate(split(tir, ratio)):
            cv2.imwrite(f'data/track1-A/_small_obj/tir/{stem}_{i}.jpg', mat)
