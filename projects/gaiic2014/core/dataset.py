from torch.utils.data import Dataset
import copy
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union, Optional
from pathlib import Path
import numpy as np
import torch
import random
import cv2
from torchvision.ops import box_iou, box_convert

import mmcv
from mmcv.transforms import BaseTransform, LoadImageFromFile, RandomFlip, RandomResize, TransformBroadcaster, Normalize, Pad, LoadAnnotations, Normalize, RandomChoice, RandomChoiceResize
from mmcv.transforms.utils import cache_random_params, cache_randomness
from mmengine.hooks import LoggerHook, CheckpointHook

from mmdet.datasets import CocoDataset, MultiImageMixDataset, BaseDetDataset
from mmdet.datasets.api_wrappers import COCO
from mmdet.registry import DATASETS, MODELS
from mmdet.structures.bbox import HorizontalBoxes, autocast_box_type
from mmdet.datasets.transforms import Resize, RandomCrop, Mosaic, PackDetInputs, MixUp, PhotoMetricDistortion, RandomFlip, CutOut, RandomAffine
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

def replace_2dmat_value_at(src, loc, des):
    # print(f"{src.shape}, {des.shape}, {loc}, {src[loc[0]: loc[0]+des.shape[0], loc[1]: loc[1] + des.shape[1]].shape}")
    src[loc[0]: loc[0]+des.shape[0], loc[1]: loc[1] + des.shape[1]] = des


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
    def __init__(self, drop_keys=('dataset', ), meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction')):
        super().__init__(meta_keys)
        self.drop_keys = drop_keys

    def transform(self, results: dict) -> dict:
        for k in self.drop_keys:
            results.pop(k, None)
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
class MultiInputCopyPaste(BaseTransform):
    def __init__(self, keys=['img', 'tir'], prob=0.6, patch_size=(256, 256), intersect_thres=0.5, scales=(0.3, 1.5), individual_pipeline=[]) -> None:
        # patch_size: (w, h)
        super().__init__()
        self.keys = keys
        self.prob = prob
        self.patch_size = patch_size
        self.intersect_thres = intersect_thres
        self.scales = scales
        self.individual_pipeline = Compose(individual_pipeline)

    def get_params(self, n, shape):
        p = random.random()
        idx = np.random.choice(n)
        src_x = np.random.randint(0, shape[0] - self.patch_size[0])
        src_y = np.random.randint(0, shape[1] - self.patch_size[1])
        scale = np.random.random() * (self.scales[1] - self.scales[0]) + self.scales[0]
        size = (int(self.patch_size[0] * scale), int(self.patch_size[1]*scale))
        des_x = np.random.randint(0, shape[0] - size[0])
        des_y = np.random.randint(0, shape[1] - size[1])
        return p, idx, (src_x, src_y), scale, size, (des_x, des_y)
    
    @classmethod
    def copy_paste(cls, src, des, keys, patch_size, thres, *args):
        (x, y), scale, (dw, dh), (dx, dy) = args
        
        for key in keys:
            patch = cv2.resize(src[key][y: y+patch_size[1], x: x+patch_size[0]], (dw, dh))
            des[key][dy: dy+dh, dx: dx+dw] = patch

        des_gt_bboxes = des['gt_bboxes'].numpy()
        # compute intersect
        iarea = np.maximum(0, np.minimum(des_gt_bboxes[:, 2], dx+dw) - np.maximum(des_gt_bboxes[:, 0], dx)) * np.maximum(0, np.minimum(des_gt_bboxes[:, 3], dy+dh) - np.maximum(des_gt_bboxes[:, 1], dy))
        area = (des_gt_bboxes[:, 2] - des_gt_bboxes[:, 0]) * (des_gt_bboxes[:, 3] - des_gt_bboxes[:, 1])
        p1 = ((iarea / area) <= thres)

        src_gt_bboxes = src['gt_bboxes'].numpy()
        iarea = np.maximum(0, np.minimum(src_gt_bboxes[:, 2], x+patch_size[0]) - np.maximum(src_gt_bboxes[:, 0], x)) * np.maximum(0, np.minimum(src_gt_bboxes[:, 3], y+patch_size[0]) - np.maximum(src_gt_bboxes[:, 1], y))
        area = (src_gt_bboxes[:, 2] - src_gt_bboxes[:, 0]) * (src_gt_bboxes[:, 3] - src_gt_bboxes[:, 1])
        p2 = ((iarea / area) > thres)
        temp = src['gt_bboxes'][p2]
        temp.rescale_([scale, scale])
        temp.translate_([dx - scale*x, dy - scale*y])

        des['gt_bboxes'] = temp.cat([des['gt_bboxes'][p1], temp], 0)
        des['gt_bboxes_labels'] = np.concatenate([ des['gt_bboxes_labels'][p1], src['gt_bboxes_labels'][p2] ], 0)
        des['gt_ignore_flags'] = np.concatenate([ des['gt_ignore_flags'][p1], src['gt_ignore_flags'][p2] ], 0)
        return des

    def transform(self, results: Dict) -> Dict | Tuple[List] | None:
        """
            在原tir上随机取一块区域;
            随机缩放;
            在目标图上随机取一块相同大小的区域。复制像素值。
            目标图中删除覆盖面积大于thres的框。
            复制原图里，与区域的iou交集大于thres的box到目标图。
        """
        dataset  :BaseDataset = results.pop('dataset')
        results = self.individual_pipeline(results)
        h, w = results['img'].shape[:2]
        p, idx, *args = self.get_params(len(dataset), (w, h))
        if p >= self.prob:
            return results
        # print(p, idx, args)
        item = self.individual_pipeline(dataset.get_data_info(idx))
        results = self.copy_paste(item, results, self.keys, self.patch_size, self.intersect_thres, *args)
        results['dataset'] = dataset
        return results


@TRANSFORMS.register_module()
class RandomDropImgRegion(BaseTransform):
    def __init__(self, key='tir', prob=0.5, smear_range=(128, 350)) -> None:
        super().__init__()
        self.key = key
        self.prob = prob
        self.smear_range = smear_range

    def get_params(self, shape):
        p = random.random()
        size_x = int(np.random.random() * (self.smear_range[1] - self.smear_range[0]) + self.smear_range[0])
        size_y = int(np.random.random() * (self.smear_range[1] - self.smear_range[0]) + self.smear_range[0])
        x = np.random.randint(0, shape[0] - size_x)
        y = np.random.randint(0, shape[1] - size_y)
        return p, (x, y), (size_x, size_y)

    def transform(self, results: Dict) -> Dict | Tuple[List] | None:
        """
            随机取一块区域, 不需要保证比例
            0填充tir图像；
        """
        img = results[self.key].copy()
        p, loc, size = self.get_params(img.shape[1::-1])
        if p < self.prob:
            img[loc[1]: loc[1]+size[1], loc[0]: loc[0]+size[0]] = 0
            results[self.key] = img
        return results


@TRANSFORMS.register_module()
class BugFreeTransformBroadcaster(TransformBroadcaster):
    def transform(self, results: Dict):
        """Broadcast wrapped transforms to multiple targets."""
        results.pop('dataset', None)
        # Apply input remapping
        inputs = self._map_input(results, self.mapping)  # patch!!

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


@TRANSFORMS.register_module()
class RandomShiftOnlyImg(BaseTransform):
    def __init__(self, key='img', prob: float = 0.5, max_shift_px: int = 32) -> None:
        assert 0 <= prob <= 1
        assert max_shift_px >= 0
        self.key = key
        self.prob = prob
        self.max_shift_px = max_shift_px

    @cache_randomness
    def _random_args(self) -> float:
        random_shift_x = random.randint(-self.max_shift_px, self.max_shift_px)
        random_shift_y = random.randint(-self.max_shift_px, self.max_shift_px)
        return random.uniform(0, 1), random_shift_x, random_shift_y

    def transform(self, results: dict) -> dict:
        """Transform function to random shift images, bounding boxes.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Shift results.
        """
        p, random_shift_x, random_shift_y = self._random_args()
        if p < self.prob:
            new_x = max(0, random_shift_x)
            ori_x = max(0, -random_shift_x)
            new_y = max(0, random_shift_y)
            ori_y = max(0, -random_shift_y)
            # shift img
            img = results[self.key]
            new_img = np.zeros_like(img)
            img_h, img_w = img.shape[:2]
            new_h = img_h - np.abs(random_shift_y)
            new_w = img_w - np.abs(random_shift_x)
            new_img[new_y:new_y + new_h, new_x:new_x + new_w] = img[ori_y:ori_y + new_h, ori_x:ori_x + new_w]
            results[self.key] = new_img
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, '
        repr_str += f'max_shift_px={self.max_shift_px}, '
        return repr_str


@TRANSFORMS.register_module()
class AdaptiveHistEQU(BaseTransform):
    def __init__(self, key='img'):
        self.key = key

    def transform(self, results):
        img = results[self.key]
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
        channels = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe.apply(channels[0], channels[0])

        ycrcb = cv2.merge(channels)
        results[self.key] = cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR)
        return results


@TRANSFORMS.register_module()
class ImgBlend(BaseTransform):
    def __init__(self, fuse_keys=['img', 'tir'], alphas=[0.25, 0.75]):
        self.fuse_keys = fuse_keys
        self.alphas = alphas
        assert len(self.alphas) == len(self.fuse_keys) == 2

    def transform(self, results):
        results['img'] = cv2.addWeighted(results[self.fuse_keys[0]], self.alphas[0], results[self.fuse_keys[1]], self.alphas[1], 0)
        return results


@TRANSFORMS.register_module()
class RandomSafeCrop(BaseTransform):
    """Random crop the image & bboxes & masks.

    The absolute ``crop_size`` is sampled based on ``crop_type`` and
    ``image_size``, then the cropped results are generated.

    Required Keys:

    - img
    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_ignore_flags (bool) (optional)
    - gt_seg_map (np.uint8) (optional)

    Modified Keys:

    - img
    - img_shape
    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)
    - gt_seg_map (optional)
    - gt_instances_ids (options, only used in MOT/VIS)

    Added Keys:

    - homography_matrix

    Args:
        crop_size (tuple): The relative ratio or absolute pixels of
            (width, height).
        crop_type (str, optional): One of "relative_range", "relative",
            "absolute", "absolute_range". "relative" randomly crops
            (h * crop_size[0], w * crop_size[1]) part from an input of size
            (h, w). "relative_range" uniformly samples relative crop size from
            range [crop_size[0], 1] and [crop_size[1], 1] for height and width
            respectively. "absolute" crops from an input with absolute size
            (crop_size[0], crop_size[1]). "absolute_range" uniformly samples
            crop_h in range [crop_size[0], min(h, crop_size[1])] and crop_w
            in range [crop_size[0], min(w, crop_size[1])].
            Defaults to "absolute".
        allow_negative_crop (bool, optional): Whether to allow a crop that does
            not contain any bbox area. Defaults to False.
        recompute_bbox (bool, optional): Whether to re-compute the boxes based
            on cropped instance masks. Defaults to False.
        bbox_clip_border (bool, optional): Whether clip the objects outside
            the border of the image. Defaults to True.

    Note:
        - If the image is smaller than the absolute crop size, return the
            original image.
        - The keys for bboxes, labels and masks must be aligned. That is,
          ``gt_bboxes`` corresponds to ``gt_labels`` and ``gt_masks``, and
          ``gt_bboxes_ignore`` corresponds to ``gt_labels_ignore`` and
          ``gt_masks_ignore``.
        - If the crop does not contain any gt-bbox region and
          ``allow_negative_crop`` is set to False, skip this image.
    """

    def __init__(self,
                 crop_size: tuple,
                 crop_type: str = 'absolute',
                 allow_negative_crop: bool = False,
                 recompute_bbox: bool = False,
                 bbox_clip_border: bool = True, block_if_below=0.5, block_keys=['img', 'tir']) -> None:
        if crop_type not in [
                'relative_range', 'relative', 'absolute', 'absolute_range'
        ]:
            raise ValueError(f'Invalid crop_type {crop_type}.')
        if crop_type in ['absolute', 'absolute_range']:
            assert crop_size[0] > 0 and crop_size[1] > 0
            assert isinstance(crop_size[0], int) and isinstance(
                crop_size[1], int)
            if crop_type == 'absolute_range':
                assert crop_size[0] <= crop_size[1]
        else:
            assert 0 < crop_size[0] <= 1 and 0 < crop_size[1] <= 1
        self.crop_size = crop_size
        self.crop_type = crop_type
        self.allow_negative_crop = allow_negative_crop
        self.bbox_clip_border = bbox_clip_border
        self.recompute_bbox = recompute_bbox
        self.block_if_below = block_if_below
        self.block_keys = block_keys

    def _crop_data(self, results: dict, crop_size: Tuple[int, int],
                   allow_negative_crop: bool) -> Union[dict, None]:
        """Function to randomly crop images, bounding boxes, masks, semantic
        segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.
            crop_size (Tuple[int, int]): Expected absolute size after
                cropping, (h, w).
            allow_negative_crop (bool): Whether to allow a crop that does not
                contain any bbox area.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        assert crop_size[0] > 0 and crop_size[1] > 0
        img = results['img']
        margin_h = max(img.shape[0] - crop_size[0], 0)
        margin_w = max(img.shape[1] - crop_size[1], 0)
        offset_h, offset_w = self._rand_offset((margin_h, margin_w))
        crop_y1, crop_y2 = offset_h, offset_h + crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + crop_size[1]

        # Record the homography matrix for the RandomCrop
        homography_matrix = np.array(
            [[1, 0, -offset_w], [0, 1, -offset_h], [0, 0, 1]],
            dtype=np.float32)
        if results.get('homography_matrix', None) is None:
            results['homography_matrix'] = homography_matrix
        else:
            results['homography_matrix'] = homography_matrix @ results[
                'homography_matrix']

        # crop the image
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape[:2]

        # crop bboxes accordingly and clip to the image boundary
        if results.get('gt_bboxes', None) is not None:
            bboxes = results['gt_bboxes']
            bboxes.translate_([-offset_w, -offset_h])
            origin_area = bboxes.areas
            if self.bbox_clip_border:
                bboxes.clip_(img_shape[:2])
            after_area = bboxes.areas
            inside = bboxes.is_inside(img_shape[:2])
            below_thres = after_area / origin_area < self.block_if_below
            need_blocked = inside & below_thres
            valid_inds = inside & (~below_thres)

            may_affected = (box_iou(bboxes[valid_inds].tensor, bboxes[need_blocked].tensor)>0).any(1)
            bkps = {key: results[key].copy() for key in self.block_keys}
            for x1, y1, x2, y2 in bboxes[need_blocked].numpy().astype(int):
                for key in self.block_keys:
                    results[key][y1: y2, x1: x2] = 0
            for x1, y1, x2, y2 in bboxes[valid_inds][may_affected].numpy().astype(int):
                for key in self.block_keys:
                    results[key][y1: y2, x1: x2] = bkps[key][y1: y2, x1: x2]
            del bkps

            # If the crop does not contain any gt-bbox area and
            # allow_negative_crop is False, skip this image.
            if (not valid_inds.any() and not allow_negative_crop):
                return None

            results['gt_bboxes'] = bboxes[valid_inds]

            if results.get('gt_ignore_flags', None) is not None:
                results['gt_ignore_flags'] = \
                    results['gt_ignore_flags'][valid_inds]

            if results.get('gt_bboxes_labels', None) is not None:
                results['gt_bboxes_labels'] = \
                    results['gt_bboxes_labels'][valid_inds]

            if results.get('gt_masks', None) is not None:
                results['gt_masks'] = results['gt_masks'][
                    valid_inds.nonzero()[0]].crop(
                        np.asarray([crop_x1, crop_y1, crop_x2, crop_y2]))
                if self.recompute_bbox:
                    results['gt_bboxes'] = results['gt_masks'].get_bboxes(
                        type(results['gt_bboxes']))

            # We should remove the instance ids corresponding to invalid boxes.
            if results.get('gt_instances_ids', None) is not None:
                results['gt_instances_ids'] = \
                    results['gt_instances_ids'][valid_inds]

        # crop semantic seg
        if results.get('gt_seg_map', None) is not None:
            results['gt_seg_map'] = results['gt_seg_map'][crop_y1:crop_y2,
                                                          crop_x1:crop_x2]

        return results

    @cache_randomness
    def _rand_offset(self, margin: Tuple[int, int]) -> Tuple[int, int]:
        """Randomly generate crop offset.

        Args:
            margin (Tuple[int, int]): The upper bound for the offset generated
                randomly.

        Returns:
            Tuple[int, int]: The random offset for the crop.
        """
        margin_h, margin_w = margin
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        return offset_h, offset_w

    @cache_randomness
    def _get_crop_size(self, image_size: Tuple[int, int]) -> Tuple[int, int]:
        """Randomly generates the absolute crop size based on `crop_type` and
        `image_size`.

        Args:
            image_size (Tuple[int, int]): (h, w).

        Returns:
            crop_size (Tuple[int, int]): (crop_h, crop_w) in absolute pixels.
        """
        h, w = image_size
        if self.crop_type == 'absolute':
            return min(self.crop_size[1], h), min(self.crop_size[0], w)
        elif self.crop_type == 'absolute_range':
            crop_h = np.random.randint(
                min(h, self.crop_size[0]),
                min(h, self.crop_size[1]) + 1)
            crop_w = np.random.randint(
                min(w, self.crop_size[0]),
                min(w, self.crop_size[1]) + 1)
            return crop_h, crop_w
        elif self.crop_type == 'relative':
            crop_w, crop_h = self.crop_size
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)
        else:
            # 'relative_range'
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            crop_h, crop_w = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * crop_h + 0.5), int(w * crop_w + 0.5)

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to randomly crop images, bounding boxes, masks,
        semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            results (Union[dict, None]): Randomly cropped results, 'img_shape'
                key in result dict is updated according to crop size. None will
                be returned when there is no valid bbox after cropping.
        """
        image_size = results['img'].shape[:2]
        crop_size = self._get_crop_size(image_size)
        results = self._crop_data(results, crop_size, self.allow_negative_crop)
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(crop_size={self.crop_size}, '
        repr_str += f'crop_type={self.crop_type}, '
        repr_str += f'allow_negative_crop={self.allow_negative_crop}, '
        repr_str += f'recompute_bbox={self.recompute_bbox}, '
        repr_str += f'bbox_clip_border={self.bbox_clip_border})'
        return repr_str


@TRANSFORMS.register_module()
class DualModalCutOut(BaseTransform):
    def __init__(
        self,
        n_holes: Union[int, Tuple[int, int]],
        cutout_shape: Optional[Union[Tuple[int, int],  List[Tuple[int, int]]]] = None,
        cutout_ratio: Optional[Union[Tuple[float, float], List[Tuple[float, float]]]] = None,
        fill_in: Union[Tuple[float, float, float], Tuple[int, int, int]] = (0, 0, 0),
        modals=['img', 'tir'],
    ) -> None:

        assert (cutout_shape is None) ^ (cutout_ratio is None), \
            'Either cutout_shape or cutout_ratio should be specified.'
        assert (isinstance(cutout_shape, (list, tuple))
                or isinstance(cutout_ratio, (list, tuple)))
        if isinstance(n_holes, tuple):
            assert len(n_holes) == 2 and 0 <= n_holes[0] < n_holes[1]
        else:
            n_holes = (n_holes, n_holes)
        self.n_holes = n_holes
        self.fill_in = fill_in
        self.with_ratio = cutout_ratio is not None
        self.candidates = cutout_ratio if self.with_ratio else cutout_shape
        if not isinstance(self.candidates, list):
            self.candidates = [self.candidates]
        self.modals = modals
        if len(modals) != 2:
            print(f"[WARN] You'd better set two modals for `DualModalCutOut`.")

    def get_params(self, h, w):
        main_model = np.random.choice(self.modals)
        holes = {}
        for m in self.modals:
            n_holes = np.random.randint(self.n_holes[0], self.n_holes[1] + 1)
            tmp = []
            for _ in range(n_holes):
                index = np.random.randint(0, len(self.candidates))
                if not self.with_ratio:
                    cutout_w, cutout_h = self.candidates[index]
                else:
                    cutout_w = int(self.candidates[index][0] * w)
                    cutout_h = int(self.candidates[index][1] * h)
                x1 = np.random.randint(0, w - cutout_w)
                y1 = np.random.randint(0, h - cutout_h)

                x2 = np.clip(x1 + cutout_w, 0, w)
                y2 = np.clip(y1 + cutout_h, 0, h)
                tmp.append([x1, y1, x2, y2])
            holes[m] = tmp
        return main_model, holes

    @autocast_box_type()
    def transform(self, results: dict) -> dict:
        """Call function to drop some regions of image."""
        h, w, c = results['img'].shape
        main_modal, holes = self.get_params(h, w)
        mask = np.zeros_like(results[main_modal], dtype='bool')
        for m in self.modals:
            for x1, y1, x2, y2 in holes[m]:
                # results[m][y1:y2, x1:x2] = self.fill_in
                if m == main_modal:
                    results[m][y1:y2, x1:x2] = self.fill_in
                    mask[y1:y2, x1:x2] = True
                else:
                    tmp = mask[y1:y2, x1:x2]
                    results[m][y1:y2, x1:x2] = results[m][y1:y2, x1:x2] * tmp + (~tmp) * self.fill_in

        return results


@TRANSFORMS.register_module()
class MultiInputMixPadding(BaseTransform):
    def __init__(self, keys=['img', 'tir'], individual_pipeline=[], size: Optional[Tuple[int, int]] = None, block_if_below=0.5, pad_val=(114, 114, 114)):
        super().__init__()
        self.keys = keys
        self.individual_pipeline = Compose(individual_pipeline)
        self.size = size
        self.block_if_below = block_if_below
        self.pad_val = pad_val

    def get_params(self, w, h, n):
        d = np.random.choice(['tl', 'tr', 'bl', 'br'])
        hpad = (self.size[0], self.size[1] - h)
        vpad = (self.size[0] - w, h)
        idx = np.random.choice(n, 2)
        return d, [hpad, vpad], idx

    def crop_and_mute(self, results, offset, crop_size):
        """
        require:
            - 
            - gt_bboxes (BaseBoxes[torch.float32]) (optional)
            - gt_bboxes_labels (np.int64) (optional)
        
        modify:
            - {keys}
            - gt_bboxes (optional)
            - gt_bboxes_labels (optional)
        """
        ret = {}
        shape = crop_size[::-1]
        for k in self.keys:
            img = results[k]
            ret[k] = img[offset[1]: offset[1] + crop_size[1], offset[0]: offset[0] + crop_size[0]]
        bboxes = results['gt_bboxes']
        bboxes.translate_([-offset[0], -offset[1]])
        origin_area = bboxes.areas
        bboxes.clip_(shape)
        inside = bboxes.is_inside(shape)
        after_area = bboxes.areas
        below_thres = after_area / origin_area < self.block_if_below
        need_blocked = inside & below_thres
        valid_inds = inside & (~below_thres)

        may_affected = (box_iou(bboxes[valid_inds].tensor, bboxes[need_blocked].tensor) > 0.1).any(1)
        bkps = {key: ret[key].copy() for key in self.keys}
        for x1, y1, x2, y2 in bboxes[need_blocked].numpy().astype(int):
            for key in self.keys:
                ret[key][y1: y2, x1: x2] = 0
        for x1, y1, x2, y2 in bboxes[valid_inds][may_affected].numpy().astype(int):
            for key in self.keys:
                ret[key][y1: y2, x1: x2] = bkps[key][y1: y2, x1: x2]
        del bkps

        ret['gt_bboxes'] = bboxes[valid_inds]
        ret['gt_bboxes_labels'] = results['gt_bboxes_labels'][valid_inds]
        return ret

    def get_offset(self, input_shape, crop_size):
        crop_size = (
            min(crop_size[0], input_shape[1]),
            min(crop_size[1], input_shape[0])
        )
        ox = np.random.randint(0, input_shape[1] - crop_size[0] + 1)
        oy = np.random.randint(0, input_shape[0] - crop_size[1] + 1)
        return (ox, oy), crop_size

    def transform(self, results: Dict) -> Dict | Tuple[List] | None:
        dataset = results.pop('dataset')
        results = self.individual_pipeline(results)
        h, w = results['img'].shape[:2]
        if w < self.size[0] or h < self.size[1]:
            d, pad, idx = self.get_params(w, h, len(dataset))
            # print(d)
            
            sub_results = [self.individual_pipeline(dataset.get_data_info(i)) for i in idx]
            for i in range(len(idx)):
                offset, size = self.get_offset(sub_results[i]['img'].shape[:2], pad[i])
                sub_results[i] = self.crop_and_mute(sub_results[i], offset, size)
            for k in self.keys:
                img = np.full((*self.size[::-1], 3), self.pad_val, dtype=results[k].dtype)
                if d == 'tl':
                    replace_2dmat_value_at(img, (0, 0), results[k])
                    replace_2dmat_value_at(img, (h, 0), sub_results[0][k])
                    replace_2dmat_value_at(img, (0, w), sub_results[1][k])
                elif d == 'tr':
                    replace_2dmat_value_at(img, (0, pad[1][0]), results[k])
                    replace_2dmat_value_at(img, (h, 0), sub_results[0][k])
                    replace_2dmat_value_at(img, (0, 0), sub_results[1][k])
                elif d == 'bl':
                    replace_2dmat_value_at(img, (pad[0][1], 0), results[k])
                    replace_2dmat_value_at(img, (0, 0), sub_results[0][k])
                    replace_2dmat_value_at(img, (pad[0][1], w), sub_results[1][k])
                else:
                    replace_2dmat_value_at(img, (pad[0][1], pad[1][0]), results[k])
                    replace_2dmat_value_at(img, (0, 0), sub_results[0][k])
                    replace_2dmat_value_at(img, (pad[0][1], 0), sub_results[1][k])
                results[k] = img
            if d == 'tl':
                sub_results[0]['gt_bboxes'].translate_([0, h])
                sub_results[1]['gt_bboxes'].translate_([w, 0])
            elif d == 'tr':
                sub_results[0]['gt_bboxes'].translate_([0, h])
                results['gt_bboxes'].translate_([pad[1][0], 0])
            elif d == 'bl':
                results['gt_bboxes'].translate_([0, pad[0][1]])
                sub_results[1]['gt_bboxes'].translate_([w, pad[0][1]])
            else:
                results['gt_bboxes'].translate_([pad[1][0], pad[0][1]])
                sub_results[1]['gt_bboxes'].translate_([0, pad[0][1]])
            results['gt_bboxes'] = results['gt_bboxes'].cat([
                results['gt_bboxes'],
                sub_results[0]['gt_bboxes'],
                sub_results[1]['gt_bboxes']
            ], 0)
            results['gt_bboxes_labels'] = np.concatenate([
                results['gt_bboxes_labels'],
                sub_results[0]['gt_bboxes_labels'],
                sub_results[1]['gt_bboxes_labels']
            ], axis=0)

        results['dataset'] = dataset
        return results


@TRANSFORMS.register_module()
class RandomRotate90(BaseTransform):
    def __init__(self, prob=0.5, dir=['l', 'r']):
        """
        Required Keys:

        - img
        - gt_bboxes (BaseBoxes[torch.float32]) (optional)

        Modified Keys:

        - img
        - img_shape
        - gt_bboxes (optional)
        """
        super().__init__()
        self.prob = prob
        self.dir = dir

    @cache_randomness
    def get_params(self):
        p = np.random.random()
        d = np.random.choice(self.dir)
        return p, d
    
    def transform(self, results: Dict) -> Dict | Tuple[List] | None:
        p, d = self.get_params()
        if p < self.prob:
            ax = (0, 1)
            if d == 'l':
                k = 1
            else:
                k = -1
            results['img'] = np.rot90(results['img'], k, ax)
            shape = results['img'].shape[:2]
            results['img_shape'] = shape
            print(p, d)
            if 'gt_bboxes' in results:
                boxes = results['gt_bboxes'].cxcywh
                if d == 'l':
                    xy = torch.cat([
                        boxes[:, 1:2],
                        shape[0] - boxes[:, 0:1]
                    ], axis=1)
                else:
                    xy = torch.cat([
                        shape[1] - boxes[:, 1:2],
                        boxes[:, 0:1]
                    ], axis=1)

                results['gt_bboxes'] = HorizontalBoxes(torch.cat([xy, boxes[:, [3, 2]]], axis=1), in_mode='cxcywh')
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
        return data_info

    def prepare_data(self, idx) -> Any:
        """Get data processed by ``self.pipeline``.

        Args:
            idx (int): The index of ``data_info``.

        Returns:
            Any: Depends on ``self.pipeline``.
        """
        data_info = self.get_data_info(idx)
        data_info['dataset'] = self
        try:
            return self.pipeline(data_info)
        except Exception as e:
            return None
