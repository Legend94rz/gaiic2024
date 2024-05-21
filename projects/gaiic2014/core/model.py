from typing import Any, Callable, Dict, List, Sequence, Tuple, Union, Optional
import torch
from torch import nn
import warnings
import math
from einops import rearrange
import copy
from torch import Tensor, nn

from fairscale.nn.checkpoint import checkpoint_wrapper
from mmcv.cnn.bricks.transformer import BaseTransformerLayer, build_transformer_layer
from mmcv.ops.multi_scale_deform_attn import MultiScaleDeformableAttention, MultiScaleDeformableAttnFunction
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.models.utils.misc import samplelist_boxtype2tensor
from mmdet.models.data_preprocessors.data_preprocessor import DetDataPreprocessor
from mmdet.models.layers import inverse_sigmoid
from mmdet.models.task_modules import BaseAssigner
from mmdet.utils import OptConfigType
from mmengine.model.utils import stack_batch
from mmengine.model import BaseModule, constant_init, xavier_init
from mmengine.model import BaseModule, ModuleList
from mmengine.structures import InstanceData
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


@MODELS.register_module()
class Sparse4Dv3QueryGenerator(BaseModule):
    """Implement query generator of the Contrastive denoising (CDN) proposed in
    `DINO: DETR with Improved DeNoising Anchor Boxes for End-to-End Object
    Detection <https://arxiv.org/abs/2203.03605>`_

    Code is modified from the `official github repo
    <https://github.com/IDEA-Research/DINO>`_.

    Args:
        num_classes (int): Number of object classes.
        embed_dims (int): The embedding dimensions of the generated queries.
        num_matching_queries (int): The queries number of the matching part.
            Used for generating dn_mask.
        label_noise_scale (float): The scale of label noise, defaults to 0.5.
        box_noise_scale (float): The scale of box noise, defaults to 1.0.
        group_cfg (:obj:`ConfigDict` or dict, optional): The config of the
            denoising queries grouping, includes `dynamic`, `num_dn_queries`,
            and `num_groups`. Two grouping strategies, 'static dn groups' and
            'dynamic dn groups', are supported. When `dynamic` is `False`,
            the `num_groups` should be set, and the number of denoising query
            groups will always be `num_groups`. When `dynamic` is `True`, the
            `num_dn_queries` should be set, and the group number will be
            dynamic to ensure that the denoising queries number will not exceed
            `num_dn_queries` to prevent large fluctuations of memory. Defaults
            to `None`.
    """

    def __init__(self,
                 num_classes: int,
                 embed_dims: int,
                 assigner,
                 num_matching_queries: int,
                 label_noise_scale: float = 0.5,
                 box_noise_scale: float = 1.0,
                 group_cfg: OptConfigType = None) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.embed_dims = embed_dims
        self.num_matching_queries = num_matching_queries    # =900. `head.num_query`
        self.label_noise_scale = label_noise_scale
        self.box_noise_scale = box_noise_scale

        # prepare grouping strategy
        group_cfg = {} if group_cfg is None else group_cfg
        self.dynamic_dn_groups = group_cfg.get('dynamic', True)
        if self.dynamic_dn_groups:
            if 'num_dn_queries' not in group_cfg:
                warnings.warn("'num_dn_queries' should be set when using dynamic dn groups, use 100 as default.")
            self.num_dn_queries = group_cfg.get('num_dn_queries', 100)      # =500
            assert isinstance(self.num_dn_queries, int), f'Expected the num_dn_queries to have type int, but got {self.num_dn_queries}({type(self.num_dn_queries)}).'
        else:
            assert 'num_groups' in group_cfg, 'num_groups should be set when using static dn groups'
            self.num_groups = group_cfg['num_groups']
            assert isinstance(self.num_groups, int), f'Expected the num_groups to have type int, but got {self.num_groups}({type(self.num_groups)}).'

        # NOTE The original repo of DINO set the num_embeddings 92 for coco,
        # 91 (0~90) of which represents target classes and the 92 (91)
        # indicates `Unknown` class. However, the embedding of `unknown` class
        # is not used in the original DINO.
        # TODO: num_classes + 1 or num_classes ?
        self.label_embedding = nn.Embedding(self.num_classes, self.embed_dims)
        self.assigner :BaseAssigner = TASK_UTILS.build(assigner)

    def __call__(self, batch_data_samples: SampleList) -> tuple:
        """Generate contrastive denoising (cdn) queries with ground truth.

        Descriptions of the Number Values in code and comments:
            - num_target_total: the total target number of the input batch
              samples.
            - max_num_target: the max target number of the input batch samples.
            - num_noisy_targets: the total targets number after adding noise,
              i.e., num_target_total * num_groups * 2.
            - num_denoising_queries: the length of the output batched queries,
              i.e., max_num_target * num_groups * 2.

        NOTE The format of input bboxes in batch_data_samples is unnormalized
        (x, y, x, y), and the output bbox queries are embedded by normalized
        (cx, cy, w, h) format bboxes going through inverse_sigmoid.

        Args:
            batch_data_samples (list[:obj:`DetDataSample`]): List of the batch
                data samples, each includes `gt_instance` which has attributes
                `bboxes` and `labels`. The `bboxes` has unnormalized coordinate
                format (x, y, x, y).

        Returns:
            tuple: The outputs of the dn query generator.

            - dn_label_query (Tensor): The output content queries for denoising
              part, has shape (bs, num_denoising_queries, dim), where
              `num_denoising_queries = max_num_target * num_groups * 2`.
            - dn_bbox_query (Tensor): The output reference bboxes as positions
              of queries for denoising part, which are embedded by normalized
              (cx, cy, w, h) format bboxes going through inverse_sigmoid, has
              shape (bs, num_denoising_queries, 4) with the last dimension
              arranged as (cx, cy, w, h).
            - attn_mask (Tensor): The attention mask to prevent information
              leakage from different denoising groups and matching parts,
              will be used as `self_attn_mask` of the `decoder`, has shape
              (num_queries_total, num_queries_total), where `num_queries_total`
              is the sum of `num_denoising_queries` and `num_matching_queries`.
            - dn_meta (Dict[str, int]): The dictionary saves information about
              group collation, including 'num_denoising_queries' and
              'num_denoising_groups'. It will be used for split outputs of
              denoising and matching parts and loss calculation.
        """
        # normalize bbox and collate ground truth (gt)
        gt_labels_list = []
        gt_bboxes_list = []
        for sample in batch_data_samples:
            img_h, img_w = sample.img_shape
            bboxes = sample.gt_instances.bboxes
            factor = bboxes.new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
            bboxes_normalized = bboxes / factor
            gt_bboxes_list.append(bboxes_normalized)
            gt_labels_list.append(sample.gt_instances.labels)
        gt_labels = torch.cat(gt_labels_list)  # (num_target_total, 4)
        gt_bboxes = torch.cat(gt_bboxes_list)   # xyxy

        num_target_list = [len(bboxes) for bboxes in gt_bboxes_list]
        max_num_target = max(num_target_list)
        num_groups = self.get_num_groups(max_num_target)

        # 为当前batch生成dn query。后续转成[B, max_num_target, dim]后，训练时直接与正常query cat起来. 注意从这里开始与DINO实现不同。
        dn_label_query = self.generate_dn_label_query(gt_labels, num_groups)     # [num_target_total*2*#Group, dim] . category labels
        dn_bbox_query = self.generate_dn_bbox_query(gt_bboxes, num_groups)       # [num_target_total*2*#Group, 4] (after inverse_sigmoid), norm box [0, 1], cxcywh
        
        # The `batch_idx` saves the batch index of the corresponding sample
        # for each target, has shape (num_target_total).
        dn_label_query, dn_bbox_query, assigned_idx = self.collate_dn_queries(
            dn_label_query, dn_bbox_query, num_target_list, max_num_target,
            batch_data_samples, num_groups
        )

        attn_mask = self.generate_dn_mask(max_num_target, num_groups, device=dn_label_query.device)

        dn_meta = dict(
            num_denoising_queries=int(max_num_target * 2 * num_groups),
            num_denoising_groups=num_groups,
            assigned_idx=assigned_idx
        )

        return dn_label_query, dn_bbox_query, attn_mask, dn_meta

    def get_num_groups(self, max_num_target: int = None) -> int:
        """Calculate denoising query groups number.

        Two grouping strategies, 'static dn groups' and 'dynamic dn groups',
        are supported. When `self.dynamic_dn_groups` is `False`, the number
        of denoising query groups will always be `self.num_groups`. When
        `self.dynamic_dn_groups` is `True`, the group number will be dynamic,
        ensuring the denoising queries number will not exceed
        `self.num_dn_queries` to prevent large fluctuations of memory.

        NOTE The `num_group` is shared for different samples in a batch. When
        the target numbers in the samples varies, the denoising queries of the
        samples containing fewer targets are padded to the max length.

        Args:
            max_num_target (int, optional): The max target number of the batch
                samples. It will only be used when `self.dynamic_dn_groups` is
                `True`. Defaults to `None`.

        Returns:
            int: The denoising group number of the current batch.
        """
        if self.dynamic_dn_groups:
            assert max_num_target is not None, 'group_queries should be provided when using dynamic dn groups'
            if max_num_target == 0:
                num_groups = 1
            else:
                num_groups = self.num_dn_queries // max_num_target  # = 500 // #targets
        else:
            num_groups = self.num_groups
        if num_groups < 1:
            num_groups = 1
        return int(num_groups)

    def generate_dn_label_query(self, gt_labels: Tensor,
                                num_groups: int) -> Tensor:
        """Generate noisy labels and their query embeddings.

        The strategy for generating noisy labels is: Randomly choose labels of
        `self.label_noise_scale * 0.5` proportion and override each of them
        with a random object category label.

        NOTE Not add noise to all labels. Besides, the `self.label_noise_scale
        * 0.5` arg is the ratio of the chosen positions, which is higher than
        the actual proportion of noisy labels, because the labels to override
        may be correct. And the gap becomes larger as the number of target
        categories decreases. The users should notice this and modify the scale
        arg or the corresponding logic according to specific dataset.

        Args:
            gt_labels (Tensor): The concatenated gt labels of all samples
                in the batch, has shape (num_target_total, ) where
                `num_target_total = sum(num_target_list)`.
            num_groups (int): The number of denoising query groups.

        Returns:
            Tensor: The query embeddings of noisy labels, has shape
            (num_noisy_targets, embed_dims), where `num_noisy_targets =
            num_target_total * num_groups * 2`.
        """

        assert self.label_noise_scale > 0
        gt_labels_expand = gt_labels.repeat(2 * num_groups, 1).view(-1) # 
        p = torch.rand_like(gt_labels_expand.float())
        chosen_indice = torch.nonzero(p < (self.label_noise_scale * 0.5)).view(-1)  # Note `* 0.5`
        new_labels = torch.randint_like(chosen_indice, 0, self.num_classes)
        noisy_labels_expand = gt_labels_expand.scatter(0, chosen_indice, new_labels)
        # dn_label_query = self.label_embedding(noisy_labels_expand)      # [num_noisy_targets] --Embedding--> [num_noisy_targets, embed_dims]
        return noisy_labels_expand

    def generate_dn_bbox_query(self, gt_bboxes: Tensor,
                               num_groups: int) -> Tensor:
        """Generate noisy bboxes and their query embeddings.

        The strategy for generating noisy bboxes is as follow:

        .. code:: text

            +--------------------+
            |      negative      |
            |    +----------+    |
            |    | positive |    |
            |    |    +-----|----+------------+
            |    |    |     |    |            |
            |    +----+-----+    |            |
            |         |          |            |
            +---------+----------+            |
                      |                       |
                      |        gt bbox        |
                      |                       |
                      |             +---------+----------+
                      |             |         |          |
                      |             |    +----+-----+    |
                      |             |    |    |     |    |
                      +-------------|--- +----+     |    |
                                    |    | positive |    |
                                    |    +----------+    |
                                    |      negative      |
                                    +--------------------+

         The random noise is added to the top-left and down-right point
         positions, hence, normalized (x, y, x, y) format of bboxes are
         required. The noisy bboxes of positive queries have the points
         both within the inner square, while those of negative queries
         have the points both between the inner and outer squares.

        Besides, the length of outer square is twice as long as that of
        the inner square, i.e., self.box_noise_scale * w_or_h / 2.
        NOTE The noise is added to all the bboxes. Moreover, there is still
        unconsidered case when one point is within the positive square and
        the others is between the inner and outer squares.

        Args:
            gt_bboxes (Tensor): The concatenated gt bboxes of all samples
                in the batch, has shape (num_target_total, 4) with the last
                dimension arranged as (cx, cy, w, h) where
                `num_target_total = sum(num_target_list)`.
            num_groups (int): The number of denoising query groups.

        Returns:
            Tensor: The output noisy bboxes, which are embedded by normalized
            (cx, cy, w, h) format bboxes going through inverse_sigmoid, has
            shape (num_noisy_targets, 4) with the last dimension arranged as
            (cx, cy, w, h), where
            `num_noisy_targets = num_target_total * num_groups * 2`.
        """
        assert self.box_noise_scale > 0
        device = gt_bboxes.device

        # expand gt_bboxes as groups
        gt_bboxes_expand = gt_bboxes.repeat(2 * num_groups, 1)  # xyxy

        # obtain index of negative queries in gt_bboxes_expand
        positive_idx = torch.arange(len(gt_bboxes), dtype=torch.long, device=device)
        positive_idx = positive_idx.unsqueeze(0).repeat(num_groups, 1)  # => [#groups, len(gt_bboxes)]
        positive_idx += 2 * len(gt_bboxes) * torch.arange(num_groups, dtype=torch.long, device=device)[:, None]
        positive_idx = positive_idx.flatten()
        negative_idx = positive_idx + len(gt_bboxes)

        # determine the sign of each element in the random part of the added
        # noise to be positive or negative randomly.
        rand_sign = torch.randint_like(
            gt_bboxes_expand, low=0, high=2,
            dtype=torch.float32) * 2.0 - 1.0  # [low, high), 1 or -1, randomly

        # calculate the random part of the added noise
        rand_part = torch.rand_like(gt_bboxes_expand)  # [0, 1)
        rand_part[negative_idx] += 1.0  # pos: [0, 1); neg: [1, 2)
        rand_part *= rand_sign  # pos: (-1, 1); neg: (-2, -1] U [1, 2)

        # add noise to the bboxes
        bboxes_whwh = bbox_xyxy_to_cxcywh(gt_bboxes_expand)[:, 2:].repeat(1, 2)
        noisy_bboxes_expand = gt_bboxes_expand + torch.mul(rand_part, bboxes_whwh) * self.box_noise_scale / 2  # xyxy
        noisy_bboxes_expand = noisy_bboxes_expand.clamp(min=0.0, max=1.0)
        noisy_bboxes_expand = bbox_xyxy_to_cxcywh(noisy_bboxes_expand)

        # dn_bbox_query = inverse_sigmoid(noisy_bboxes_expand, eps=1e-3)
        return noisy_bboxes_expand

    def collate_dn_queries(self, input_label_query: Tensor,
                           input_bbox_query: Tensor, num_target_list, max_num_target,
                           batch_data_samples: SampleList, num_groups: int) -> Tuple[Tensor]:
        """
        args:
            input_label_query: [\sum num_target_list * 2 * #group, ]
            input_bbox_query: [\sum num_target_list * 2 * #group, 4]

        returns:
            dn_label_query: [b, max_t * 2 * #group, dim]. after label embedding.
            dn_bbox_query:  [b, max_t * 2 * #group, 4]. after `inverse_sigmoid`
            assigned_idx: LongTensor. [b, max_t*2*#group]. pad -1
        """
        batch_size = len(batch_data_samples)
        dn_label_query = torch.zeros((batch_size, max_num_target * 2 * num_groups, self.embed_dims), device=input_label_query.device)
        dn_bbox_query = torch.zeros((batch_size, max_num_target * 2 * num_groups, 4), device=input_bbox_query.device)
        assigned_idx = torch.full((batch_size, max_num_target * 2 * num_groups), -1)

        input_label_query = torch.chunk(input_label_query, num_groups)
        input_bbox_query = torch.chunk(input_bbox_query, num_groups)
        qsize = [x for x in num_target_list]*2
        for i in range(num_groups):
            # input_label_query[i]: 第i组的 正负query. [\sum num_target_list * 2, ]. 离散类别, [0, self.num_classes)
            # input_bbox_query[i] : normed bbox [0, 1]. [\sum num_target_list * 2, 4] cxcywh
            lq = torch.split(input_label_query[i], qsize)
            bq = torch.split(input_bbox_query[i], qsize)
            lq = [torch.cat([lq[j], lq[j+batch_size]]) for j in range(batch_size)]
            bq = [torch.cat([bq[j], bq[j+batch_size]]) for j in range(batch_size)]
            for j in range(batch_size):
                img_h, img_w = batch_data_samples[j].img_shape
                factor = bq[j].new_tensor([img_w, img_h, img_w, img_h]).unsqueeze(0)
                cls_score = nn.functional.one_hot(lq[j], self.num_classes)
                box_query = bbox_cxcywh_to_xyxy(bq[j]) * factor
                pred_instances = InstanceData(scores=cls_score, bboxes=box_query)
                assign = self.assigner.assign(pred_instances, batch_data_samples[j].gt_instances, batch_data_samples[j].metainfo)
                gt_inds = assign.gt_inds  # 0 表示没匹配到的。 >0 表示匹配到的 gt_instances 索引+1. shape: (num_target_list[j] * 2)
                k = max_num_target*2* i
                dn_label_query[j, k: k + len(gt_inds)] = self.label_embedding(lq[j])
                dn_bbox_query[j, k: k + len(gt_inds)] = inverse_sigmoid(bq[j], eps=1e-3)
                assigned_idx[j, k: k + len(gt_inds)] = gt_inds.cpu()
        return dn_label_query, dn_bbox_query, assigned_idx

    def generate_dn_mask(self, max_num_target: int, num_groups: int,
                         device: Union[torch.device, str]) -> Tensor:
        """Generate attention mask to prevent information leakage from
        different denoising groups and matching parts.

        .. code:: text

                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        0 0 0 0 1 1 1 1 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 0 0 0 0 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
                        1 1 1 1 1 1 1 1 0 0 0 0 0
         max_num_target |_|           |_________| num_matching_queries
                        |_____________| num_denoising_queries

               1 -> True  (Masked), means 'can not see'.
               0 -> False (UnMasked), means 'can see'.

        Args:
            max_num_target (int): The max target number of the input batch
                samples.
            num_groups (int): The number of denoising query groups.
            device (obj:`device` or str): The device of generated mask.

        Returns:
            Tensor: The attention mask to prevent information leakage from
            different denoising groups and matching parts, will be used as
            `self_attn_mask` of the `decoder`, has shape (num_queries_total,
            num_queries_total), where `num_queries_total` is the sum of
            `num_denoising_queries` and `num_matching_queries`.
        """
        num_denoising_queries = int(max_num_target * 2 * num_groups)
        num_queries_total = num_denoising_queries + self.num_matching_queries
        attn_mask = torch.zeros(
            num_queries_total,
            num_queries_total,
            device=device,
            dtype=torch.bool)
        # Make the matching part cannot see the denoising groups
        attn_mask[num_denoising_queries:, :num_denoising_queries] = True
        # Make the denoising groups cannot see each other
        for i in range(num_groups):
            # Mask rows of one group per step.
            row_scope = slice(max_num_target * 2 * i,
                              max_num_target * 2 * (i + 1))
            left_scope = slice(max_num_target * 2 * i)
            right_scope = slice(max_num_target * 2 * (i + 1),
                                num_denoising_queries)
            attn_mask[row_scope, right_scope] = True
            attn_mask[row_scope, left_scope] = True
        return attn_mask

