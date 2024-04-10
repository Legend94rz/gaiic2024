from torch.utils.data import Dataset
import copy
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union
from pathlib import Path
import numpy as np
import torch
import random

import mmcv
from mmcv.transforms import BaseTransform, LoadImageFromFile, RandomFlip, RandomResize, TransformBroadcaster, Normalize, Pad, LoadAnnotations, Resize, Normalize
from mmcv.transforms.utils import cache_random_params

from mmdet.datasets import CocoDataset, MultiImageMixDataset, BaseDetDataset
from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import DATASETS, MODELS
from mmdet.datasets.transforms import RandomCrop, Mosaic, PackDetInputs, MixUp, PhotoMetricDistortion, RandomFlip
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor

from mmengine.registry import TRANSFORMS
from mmengine.model.utils import stack_batch
import mmengine.fileio as fileio
from mmengine.dataset import Compose, BaseDataset

"""
crop/resize/flip => [img & tir]
hsv/jitter/etc. -> img / tir
=======
Train pipeline:
LoadRGB
LoadTir
LoadAnno
RandomCropReszie(RGBT): resize(0.7~1.3x), crop()
RandomFlip(RGBT)
??RandomPerspective(Translation/Rotation)
Color / HSV / bright / Contrast etc.
Mosaic(max_size=(1024+512)) with prob

=======
val & test pipeline:
LoadRGB
LoadTir
LoadAnno
Resize(RGBT) to slightly larger
TTA: [Color / HSV / bright / Contrast etc.]

"""


@TRANSFORMS.register_module()
class LoadTirFromPath(LoadImageFromFile):
    def transform(self, results: dict) -> dict | None:
        filename = results['tir_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype('float32')

        results['tir'] = img
        return results


@TRANSFORMS.register_module()
class CustomPackDetInputs(PackDetInputs):   
    def transform(self, results: dict) -> dict:
        packed_results = super().transform(results)
        packed_results['tir'] = torch.tensor(np.ascontiguousarray(results['tir'])).permute(2, 0, 1)
        return packed_results


@TRANSFORMS.register_module()
class MultiInputMosaic(BaseTransform):
    def __init__(self, keys=['img', 'tir'], prob=1.0, img_scale: Tuple[int, int] = (640, 640), center_ratio_range: Tuple[float, float] = (0.5, 1.5), pad_val: float = 114.0, bbox_clip_border: bool = True, individual_pipeline=[]) -> None:
        # img_scale: [w, h]
        super().__init__()
        self.keys = keys
        self.prob = prob
        self.img_scale = img_scale
        self.center_ratio_range = center_ratio_range
        self.pad_val = pad_val
        self.bbox_clip_border = bbox_clip_border
        self.individual_pipeline = Compose(individual_pipeline)

    def get_params(self, n):
        idx = np.random.choice(n, 3)
        p = np.random.rand()
        center_x = int(random.uniform(*self.center_ratio_range) * self.img_scale[0])
        center_y = int(random.uniform(*self.center_ratio_range) * self.img_scale[1])
        return idx, p, (center_x, center_y)
    
    def transform(self, results: Dict) -> Dict | Tuple[List] | None:
        dataset :BaseDataset = results.pop('dataset')  # avoid deep copy in the individual pipeline
        idx, p, center = self.get_params(len(dataset))
        results = self.individual_pipeline(results)
        if p < self.prob:
            others = [self.individual_pipeline(dataset.get_data_info(i)) for i in idx]
            results = self.combine(
                self.keys,
                [results] + others,
                center,
                self.img_scale, self.pad_val, self.bbox_clip_border
            )
        results['dataset'] = dataset # re-add dataset
        return results

    @classmethod
    def get_coord(cls, loc, center, target_shape, img_shape):
        # img_shape: [w, h]
        if loc == 'top_left':
            # index0 to top left part of image
            x1, y1, x2, y2 = max(center[0] - img_shape[0], 0), \
                             max(center[1] - img_shape[1], 0), \
                             center[0], center[1]
            crop_coord = img_shape[0] - (x2 - x1), img_shape[1] - (
                y2 - y1), img_shape[0], img_shape[1]
        elif loc == 'top_right':
            # index1 to top right part of image
            x1, y1, x2, y2 = center[0], \
                             max(center[1] - img_shape[1], 0), \
                             min(center[0] + img_shape[0], target_shape[0] * 2), \
                             center[1]
            crop_coord = 0, img_shape[1] - (y2 - y1), min(img_shape[0], x2 - x1), img_shape[1]
        elif loc == 'bottom_left':
            # index2 to bottom left part of image
            x1, y1, x2, y2 = max(center[0] - img_shape[0], 0), \
                             center[1], center[0], \
                             min(target_shape[1] * 2, center[1] + img_shape[1])
            crop_coord = img_shape[0] - (x2 - x1), 0, img_shape[0], min(y2 - y1, img_shape[1])
        else:
            # index3 to bottom right part of image
            x1, y1, x2, y2 = center[0], center[1], min(center[0] + img_shape[0], target_shape[0] * 2), min(target_shape[1] * 2, center[1] + img_shape[1])
            crop_coord = 0, 0, min(img_shape[0], x2 - x1), min(y2 - y1, img_shape[1])
        paste_coord = x1, y1, x2, y2
        return paste_coord, crop_coord

    @classmethod
    def combine(cls, keys, results_list, center, target_size, pad_val, bbox_clip_border):
        # target_size: [w, h]
        mosaic_img = {}
        for k in keys:
            img0 = results_list[0][k]
            if len(img0.shape) == 3:
                mosaic_img[k] = np.full((int(target_size[1] * 2), int(target_size[0] * 2), 3), pad_val, dtype=img0.dtype)
            else:
                mosaic_img[k] = np.full((int(target_size[1] * 2), int(target_size[0] * 2)), pad_val, dtype=img0.dtype)
        assert len({tuple(mosaic_img[k].shape[:2]) for k in keys}) == 1, f"The image shape from `{keys}` are different!"

        results = results_list[0]
        loc_strs = ('top_left', 'top_right', 'bottom_left', 'bottom_right')
        mosaic_bboxes = []
        mosaic_bboxes_labels = []
        mosaic_ignore_flags = []
        for i, loc in enumerate(loc_strs):
            for k in keys:
                img_i = results_list[i][k].copy()
                h_i, w_i = img_i.shape[:2]

                # keep_ratio resize
                scale_ratio_i = min(target_size[1] / h_i, target_size[0] / w_i)
                img_i = mmcv.imresize(img_i, (int(w_i * scale_ratio_i), int(h_i * scale_ratio_i)))

                # compute the combine parameters
                paste_coord, crop_coord = cls.get_coord(loc, center, target_size, img_i.shape[:2][::-1])
                x1_p, y1_p, x2_p, y2_p = paste_coord
                x1_c, y1_c, x2_c, y2_c = crop_coord

                # crop and paste image
                mosaic_img[k][y1_p:y2_p, x1_p:x2_p] = img_i[y1_c:y2_c, x1_c:x2_c]

            # adjust coordinate
            gt_bboxes_i = copy.deepcopy(results_list[i]['gt_bboxes'])
            gt_bboxes_labels_i = copy.deepcopy(results_list[i]['gt_bboxes_labels'])
            gt_ignore_flags_i = copy.deepcopy(results_list[i]['gt_ignore_flags'])
            
            padw = x1_p - x1_c
            padh = y1_p - y1_c
            gt_bboxes_i.rescale_([scale_ratio_i, scale_ratio_i])
            gt_bboxes_i.translate_([padw, padh])
            mosaic_bboxes.append(gt_bboxes_i)
            mosaic_bboxes_labels.append(gt_bboxes_labels_i)
            mosaic_ignore_flags.append(gt_ignore_flags_i)
        mosaic_bboxes = mosaic_bboxes[0].cat(mosaic_bboxes, 0)
        mosaic_bboxes_labels = np.concatenate(mosaic_bboxes_labels, 0)
        mosaic_ignore_flags = np.concatenate(mosaic_ignore_flags, 0)
        if bbox_clip_border:
            mosaic_bboxes.clip_([2 * target_size[1], 2 * target_size[0]])
        # remove outside bboxes
        inside_inds = mosaic_bboxes.is_inside([2 * target_size[1], 2 * target_size[0]]).numpy()
        mosaic_bboxes = mosaic_bboxes[inside_inds]
        mosaic_bboxes_labels = mosaic_bboxes_labels[inside_inds]
        mosaic_ignore_flags = mosaic_ignore_flags[inside_inds]

        for k in keys:
            results[k] = mosaic_img[k]
            results['img_shape'] = mosaic_img[k].shape[:2]
        results['gt_bboxes'] = mosaic_bboxes
        results['gt_bboxes_labels'] = mosaic_bboxes_labels
        results['gt_ignore_flags'] = mosaic_ignore_flags
        return results


@TRANSFORMS.register_module()
class BugFreeTransformBroadcaster(TransformBroadcaster):
    def transform(self, results: Dict):
        """Broadcast wrapped transforms to multiple targets."""

        # Apply input remapping
        inputs = self._map_input(copy.deepcopy(results), self.mapping)  # patch!!

        # Scatter sequential inputs into a list
        input_scatters = self.scatter_sequence(inputs)

        # Control random parameter sharing with a context manager
        if self.share_random_params:
            # The context manager :func`:cache_random_params` will let
            # cacheable method of the transforms cache their outputs. Thus
            # the random parameters will only generated once and shared
            # by all data items.
            ctx = cache_random_params  # type: ignore
        else:
            ctx = nullcontext  # type: ignore

        with ctx(self.transforms):
            output_scatters = [
                self._apply_transforms(copy.deepcopy(_input)) for _input in input_scatters  # patch!!
            ]

        # Collate output scatters (list of dict to dict of list)
        outputs = {
            key: [_output[key] for _output in output_scatters]
            for key in output_scatters[0]
        }
        
        # Apply remapping
        outputs = self._map_output(outputs, self.remapping)

        results.update(outputs)
        return results


@DATASETS.register_module()
class GAIIC2014DatasetV2(CocoDataset):
    METAINFO = {
        'classes': ('car', 'truck', 'bus', 'van', 'freight_car'),
        'palette': [(5, 5, 214), (26, 237, 26), (225, 10, 10), (32, 244, 244), (230, 18, 230)]
    }

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``

        Returns:
            Union[dict, List[dict]]: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}
        data_info['img_path'] = Path(self.data_prefix['img_path']) / img_info['file_name']
        data_info['tir_path'] = Path(self.data_prefix['tir_path']) / img_info['file_name']
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = None
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['classes']
            data_info['caption_prompt'] = self.caption_prompt
            data_info['custom_entities'] = True

        instances = []
        for i, ann in enumerate(ann_info):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            if ann.get('iscrowd', False):
                instance['ignore_flag'] = 1
            else:
                instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = self.cat2label[ann['category_id']]

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            instances.append(instance)
        data_info['instances'] = instances
        data_info['dataset'] = self
        return data_info


