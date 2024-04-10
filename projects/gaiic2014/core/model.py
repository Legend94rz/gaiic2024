from typing import Any, Callable, Dict, List, Sequence, Tuple, Union, Optional
import torch
from torch import nn
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention
from mmdet.registry import  MODELS
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from mmengine.model.utils import stack_batch


@MODELS.register_module()
class CustomePreprocessor(DetDataPreprocessor):
    def stack_and_norm(self, inputs):
        batch_inputs = []
        for _batch_input in inputs:
            # channel transform: rgb <--> bgr
            if self._channel_conversion:
                _batch_input = _batch_input[[2, 1, 0], ...]
            # Convert to float after channel conversion to ensure efficiency
            _batch_input = _batch_input.float()
            # Normalization.
            _batch_input = (_batch_input - self.mean) / self.std
            batch_inputs.append(_batch_input)
        # Pad and stack Tensor.
        return stack_batch(batch_inputs, self.pad_size_divisor, self.pad_value)

    def forward(self, data: Dict, training: bool = False) -> Dict:
        data = self.cast_data(data)
        data.setdefault('data_samples', None)
        inputs, data_samples = data['inputs'], data['data_samples']

        inputs = self.stack_and_norm(inputs)
        tir = self.stack_and_norm(data['tir'])

        if data_samples is not None:
            batch_pad_shape = self._get_pad_shape(data)
            # NOTE the batched image size information may be useful, e.g.
            # in DETR, this is needed for the construction of masks, which is
            # then used for the transformer_head.
            batch_input_shape = tuple(inputs[0].size()[-2:])
            for data_sample, pad_shape in zip(data_samples, batch_pad_shape):
                data_sample.set_metainfo({
                    'batch_input_shape': batch_input_shape,
                    'pad_shape': pad_shape
                })

            if self.boxtype2tensor:
                samplelist_boxtype2tensor(data_samples)

            if self.pad_mask and training:
                self.pad_gt_masks(data_samples)

            if self.pad_seg and training:
                self.pad_gt_sem_seg(data_samples)

        return {'inputs': inputs, 'tir': tir, 'data_samples': data_samples}


@MODELS.register_module()
class FuseMSDeformAttention(MultiScaleDeformableAttention):
    def __init__(self, embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_points: int = 4,
                 im2col_step: int = 64,
                 dropout: float = 0.1,
                 batch_first: bool = False,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 value_proj_ratio: float = 1.0):
        super().__init__(embed_dims, num_heads, num_levels, num_points, im2col_step, dropout, batch_first, norm_cfg, init_cfg, value_proj_ratio)
        self.agg = nn.Sequential(
            nn.Conv1d(2, 32, 1),
            nn.ReLU(),
            nn.Conv1d(32, 1, 1)
        )

    def forward(self,
                query: torch.Tensor,
                _: Optional[torch.Tensor] = None,
                value: Optional[torch.Tensor] = None,
                identity: Optional[torch.Tensor] = None,
                query_pos: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None,
                reference_points: Optional[torch.Tensor] = None,
                spatial_shapes: Optional[torch.Tensor] = None,
                level_start_index: Optional[torch.Tensor] = None,
                **kwargs) -> torch.Tensor:
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (torch.Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            value (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (torch.Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (torch.Tensor): The positional encoding for `query`.
                Default: None.
            key_padding_mask (torch.Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            reference_points (torch.Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            spatial_shapes (torch.Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (torch.Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        kwargs:
            value2: (torch.Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """
        # repeat: query / [identity] / [query_pos] / key_padding_mask / reference_points
        # stack: value 1 & 2
        if kwargs.get('value2') is not None:
            nq, bs, _ = query.shape
            query = torch.cat([query, query], dim=1)
            if identity is not None:
                identity = torch.cat([identity, identity], dim=1)
            if query_pos is not None:
                query_pos = torch.cat([query_pos, query_pos], dim=1)
            key_padding_mask = torch.cat([key_padding_mask, key_padding_mask], dim=0)
            reference_points = torch.cat([reference_points, reference_points], dim=0)
            value = torch.cat([value, kwargs['value2']], dim=1)

            hs = super().forward(query, _, value, identity, query_pos, reference_points, spatial_shapes, level_start_index, **kwargs)
            hs = hs.reshape(2, bs, nq, -1).permute(1, 0, 2, 3)  # => [b, 2, q, c]
            hs = self.agg(hs).squeeze(1).permute(1, 0, 2)       # => [b, 1, q, c] => [b, q, c] => [q, b, c]
        else:
            hs = super().forward(query, _, value, identity, query_pos, reference_points, spatial_shapes, level_start_index, **kwargs)

        return hs
