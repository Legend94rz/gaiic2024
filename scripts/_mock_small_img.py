import json
import argparse
import cv2
from pathlib import Path
import numpy as np
from itertools import chain
from mmdet.registry import DATASETS, MODELS
from mmdet.utils import register_all_modules
from projects.gaiic2014.core import *

save_dir = Path("data/track1-A/mock_rescale")
image_size = (640, 512)
transform_broadcast = dict(
    type="BugFreeTransformBroadcaster",
    mapping={
        "img": ["tir", "img"],
        "img_shape": ["img_shape", "img_shape"],
        "gt_bboxes": ['gt_bboxes', 'gt_bboxes'],
        "gt_bboxes_labels": ['gt_bboxes_labels', 'gt_bboxes_labels'],
        "gt_ignore_flags": ['gt_ignore_flags', 'gt_ignore_flags'],
        "scale_factor": ['scale_factor', 'scale_factor'],
        "flip": ["flip", "flip"],
        "flip_direction": ["flip_direction", "flip_direction"],
    },
    auto_remap=True,
    share_random_params=True,
)
train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadTirFromPath"),
    dict(type="LoadAnnotations", with_bbox=True),

    dict(type="RandomChoice", transforms=[
        [dict(**transform_broadcast, transforms=[dict(type='Resize', scale_factor=0.25, interpolation='nearest')])],
        [dict(**transform_broadcast, transforms=[dict(type='Resize', scale_factor=0.25, interpolation='bilinear')])],
        [dict(**transform_broadcast, transforms=[dict(type='Resize', scale_factor=0.25, interpolation='bicubic')])],
        [dict(**transform_broadcast, transforms=[dict(type='Resize', scale_factor=0.25, interpolation='area')])],
        [dict(**transform_broadcast, transforms=[dict(type='Resize', scale_factor=0.25, interpolation='lanczos')])],
    ]),
    dict(**transform_broadcast, transforms=[dict(type='Resize', scale=image_size)]),
]

dscfg = dict(
        type='GAIIC2014DatasetV2',
        data_root='data/track1-A',
        serialize_data=False,
        ann_file="annotations/train_updated.json",
        data_prefix=dict(img_path="train/rgb", tir_path="train/tir"),
        # filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
    )


if __name__ == "__main__":
    register_all_modules()
    ds = DATASETS.build(dscfg)
    rgb = save_dir / "rgb"
    tir = save_dir / "tir"
    rgb.mkdir(exist_ok=True, parents=True)
    tir.mkdir(exist_ok=True, parents=True)
    for i in np.random.choice(len(ds), 100, replace=False):
        item = ds[i]
        cv2.imwrite(str(rgb / Path(item['img_path']).name), item['img'])
        cv2.imwrite(str(tir / Path(item['tir_path']).name), item['tir'])
