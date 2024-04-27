from typing import Any, Callable, Dict, List, Sequence, Tuple, Union, Optional
import torch
from torch import nn
import warnings
import math
from einops import rearrange
import copy
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, build_transformer_layer
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention, MultiScaleDeformableAttnFunction
from mmdet.registry import MODELS
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from mmengine.model.utils import stack_batch
from mmengine.model import BaseModule, constant_init, xavier_init
from mmengine.model import BaseModule, ModuleList
from fairscale.nn.checkpoint import checkpoint_wrapper
from mmcv.cnn import build_norm_layer


@MODELS.register_module()
class CustomPreprocessor(DetDataPreprocessor):
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
        # self.agg = nn.Sequential(
        #     nn.Conv2d(2, 32, 1),
        #     nn.ReLU(),
        #     nn.Conv2d(32, 1, 1)
        # )

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

        #### INCORRECT:
        # nq, bs, _ = query.shape
        # query = torch.cat([query, query], dim=1)
        # if identity is not None:
        #     identity = torch.cat([identity, identity], dim=1)
        # if query_pos is not None:
        #     query_pos = torch.cat([query_pos, query_pos], dim=1)
        # key_padding_mask = torch.cat([key_padding_mask, key_padding_mask], dim=0)
        # reference_points = torch.cat([reference_points, reference_points], dim=0)
        # value = torch.cat([value, kwargs['value2']], dim=1)

        # hs = super().forward(query, _, value, identity, query_pos, key_padding_mask, reference_points, spatial_shapes, level_start_index, **kwargs)
        # hs = hs.reshape(2, bs, nq, -1).permute(1, 0, 2, 3)  # => [b, 2, q, c]
        # # hs = self.agg(hs).squeeze(1).permute(1, 0, 2)       # => [b, 1, q, c] => [b, q, c] => [q, b, c]
        # hs = ((hs[:, 0] + hs[:, 1]) / 2).permute(1, 0, 2)

        #### CORRECT:
        hs0 = super().forward(query, _, query, identity, query_pos, key_padding_mask, reference_points, spatial_shapes, level_start_index, **kwargs)
        hs1 = super().forward(query, _, kwargs['value2'], identity, query_pos, key_padding_mask, reference_points, spatial_shapes, level_start_index, **kwargs)
        return (hs0 + hs1) / 2



@MODELS.register_module()
class DualModalDeformableAttention(BaseModule):
    """An attention module used in Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 8.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        value_proj_ratio (float): The expansion ratio of value_proj.
            Default: 1.0.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 num_levels: int = 4,
                 num_points: int = 4,
                 im2col_step: int = 64,
                 dropout: float = 0.1,
                 batch_first: bool = False,
                 norm_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 value_proj_ratio: float = 1.0):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.n_modal = 2
        self.sampling_offsets = nn.Linear(embed_dims, num_heads * self.n_modal * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims, num_heads * self.n_modal * num_levels * num_points)
        value_proj_size = int(embed_dims * value_proj_ratio)
        self.value_proj = nn.Linear(embed_dims, value_proj_size)
        self.output_proj = nn.Linear(value_proj_size, embed_dims)
        self.init_weights()

    def init_weights(self) -> None:
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        device = next(self.parameters()).device
        thetas = torch.arange(
            self.num_heads, dtype=torch.float32,
            device=device) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
                         self.num_heads, 1, 1,
                         2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1).repeat(self.n_modal)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query: torch.Tensor,
                key: Optional[torch.Tensor] = None,
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
            key (torch.Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
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

        Returns:
            torch.Tensor: forwarded results with shape
            [num_query, bs, embed_dims].
        """
        # stack: spatial_shapes, level_start_index, reference_points, key_padding_mask, query_pos
        # spatial_shapes = spatial_shapes.repeat(2, 1)
        # level_start_index = torch.cat([level_start_index, level_start_index + spatial_shapes.prod(-1).sum()])
        # reference_points = reference_points.repeat(1, 2, 2, 1)   # NOTE: requires no grad
        # key_padding_mask = key_padding_mask.repeat(1, 2)
        # query_pos = query_pos.repeat(2, 1, 1)

        level_start_index = torch.cat([level_start_index, level_start_index + spatial_shapes.prod(-1).sum()])
        spatial_shapes = spatial_shapes.repeat(2, 1)
        reference_points = reference_points.repeat(1, 2, 2, 1)   # NOTE: requires no grad
        key_padding_mask = key_padding_mask.repeat(1, 2)
        query_pos = query_pos.repeat(2, 1, 1)
        
        if value is None:
            value = query

        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        #sampling_offsets = self.sampling_offsets(query).view(
        #    bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        #attention_weights = self.attention_weights(query).view(
        #    bs, num_query, self.num_heads, self.num_levels * self.num_points)
        sampling_offsets = rearrange(self.sampling_offsets(query), 'b q (h m l p o) -> b q h (m l) p o', m=2, h=self.num_heads, l=self.num_levels, p=self.num_points, o=2)
        attention_weights = rearrange(self.attention_weights(query), 'b q (h m l p) -> b q h m (l p)', m=2, h=self.num_heads, l=self.num_levels, p=self.num_points)
        attention_weights = attention_weights.softmax(-1)
        attention_weights = rearrange(attention_weights, 'b q h m (l p) -> b q h (m l) p', m=2, h=self.num_heads, l=self.num_levels, p=self.num_points)

        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] \
                + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] \
                + sampling_offsets / self.num_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')
        output = MultiScaleDeformableAttnFunction.apply(
            value, spatial_shapes, level_start_index, sampling_locations,
            attention_weights, self.im2col_step)

        output = self.output_proj(output)

        if not self.batch_first:
            # (num_query, bs ,embed_dims)
            output = output.permute(1, 0, 2)

        return self.dropout(output) + identity


@MODELS.register_module()
class DualModalEncoder(BaseModule):
    """TransformerEncoder of DETR.

    Args:
        post_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`. Only used when `self.pre_norm` is `True`
    """

    def __init__(self,
                 transformerlayers=None, num_layers=None, init_cfg=None,
                 post_norm_cfg=dict(type='LN'),
                 with_cp=-1,
                 **kwargs):
        super().__init__(init_cfg)
        if isinstance(transformerlayers, dict):
            transformerlayers = [
                copy.deepcopy(transformerlayers) for _ in range(num_layers)
            ]
        else:
            assert isinstance(transformerlayers, list) and \
                   len(transformerlayers) == num_layers
        self.num_layers = num_layers
        self.layers = ModuleList()
        for i in range(num_layers):
            self.layers.append(build_transformer_layer(transformerlayers[i]))
        self.embed_dims = self.layers[0].embed_dims
        self.pre_norm = self.layers[0].pre_norm
        #=========super=============

        if post_norm_cfg is not None:
            self.post_norm = build_norm_layer(
                post_norm_cfg, self.embed_dims)[1] if self.pre_norm else None
        else:
            assert not self.pre_norm, f'Use prenorm in ' \
                                      f'{self.__class__.__name__},' \
                                      f'Please specify post_norm_cfg'
            self.post_norm = None
        self.with_cp = with_cp
        if self.with_cp > 0:
            if checkpoint_wrapper is None:
                warnings.warn('If you want to reduce GPU memory usage, \
                              please install fairscale by executing the \
                              following command: pip install fairscale.')
                return
            for i in range(self.with_cp):
                self.layers[i] = checkpoint_wrapper(self.layers[i])

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                value2=None,
                **kwargs):
        """Forward function for `TransformerCoder`.

        Args:
            query (Tensor): Input query with shape
                `(num_queries, bs, embed_dims)`.
            key (Tensor): The key tensor with shape
                `(num_keys, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_keys, bs, embed_dims)`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor], optional): Each element is 2D Tensor
                which is used in calculation of corresponding attention in
                operation_order. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in self-attention
                Default: None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor:  results with shape [num_queries, bs, embed_dims].
        """
        assert key is None and value is None
        nout = len(query)
        query = torch.cat([query, value2], dim=0)
        for layer in self.layers:
            query = layer(
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                attn_masks=attn_masks,
                query_key_padding_mask=query_key_padding_mask,
                key_padding_mask=key_padding_mask,
                **kwargs)
        query = query[:nout]
        return query
