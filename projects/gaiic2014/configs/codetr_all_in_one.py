default_scope = "mmdet"

custom_imports = dict(
    imports=[
        "projects.CO-DETR.codetr",
        "projects.gaiic2014.core",
    ],
    allow_failed_imports=False,
)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    logger=dict(type="LoggerHook", interval=50),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1, by_epoch=True, max_keep_ckpts=3),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
    ema=dict(type="MyEMAHook", ema_type="StochasticWeightAverage", begin_epoch=9)
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method="fork", opencv_num_threads=0),
    dist_cfg=dict(backend="nccl"),
)

vis_backends = [
    dict(type="LocalVisBackend"),
    dict(
        type="WandbVisBackend",
        save_dir='wandb',
        init_kwargs=dict(
            project="gaiic2014",
            save_code=True,
            settings={"code_dir": "projects"},
            notes="",
            tags=[]
        ),
    )
]
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
backend_args = None

resume = False
num_dec_layer = 6
loss_lambda = 2.0
num_classes = 5
dataset_type = "GAIIC2014DatasetV2"
data_root = "data/track1-A/"
pretrained = "ckpt/swin_large_patch4_window12_384_22k.pth"
load_from = "ckpt/co_dino_5scale_swin_large_16e_o365tococo-614254c9_patched.pth"
# load_from = "ckpt/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth"

# image_size = (1024, 1024)
# batch_augments = [
#     dict(type='BatchFixedSizePad', size=image_size, pad_mask=True)
# ]

# model settings
model = dict(
    type="CoDETR",
    # If using the lsj augmentation,
    # it is recommended to set it to True.
    use_lsj=False,
    backbone=dict(
        type="SwinTransformer",
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        # Please only add indices that would be used
        # in FPN, otherwise some parameter will not be used
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type="Pretrained", checkpoint=pretrained),
    ),
    # detr: 52.1
    # one-stage: 49.4
    # two-stage: 47.9
    eval_module="detr",  # in ['detr', 'one-stage', 'two-stage']
    # data_preprocessor=dict(
    #     type="DetDataPreprocessor",
    #     mean=[123.675, 116.28, 103.53],
    #     std=[58.395, 57.12, 57.375],
    #     bgr_to_rgb=True,
    #     pad_mask=False,
    #     batch_augments=None,
    # ),
    data_preprocessor=dict(
        type="CustomPreprocessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
        batch_augments=None,
    ),
    neck=dict(
        type="ChannelMapper",
        in_channels=[192, 384, 768, 1536],
        kernel_size=1,
        out_channels=256,
        act_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32),
        num_outs=5,
    ),
    query_head=dict(
        type="CoDINOHead",
        num_query=900,
        num_classes=num_classes,
        in_channels=2048,
        as_two_stage=True,
        dn_cfg=dict(
            label_noise_scale=0.5,
            box_noise_scale=0.4,
            group_cfg=dict(num_dn_queries=500),
        ),
        transformer=dict(
            type="CoDinoTransformer",
            with_coord_feat=False,
            num_co_heads=2,  # ATSS Aux Head + Faster RCNN Aux Head
            num_feature_levels=5,
            encoder=dict(
                # type="DetrTransformerEncoder",
                type="DualModalEncoder",
                num_layers=6,
                # number of layers that use checkpoint.
                # The maximum value for the setting is num_layers.
                # FairScale must be installed for it to work.
                with_cp=6,
                transformerlayers=dict(
                    type="BaseTransformerLayer",
                    attn_cfgs=dict(
                        type="DualModalDeformableAttention",
                        # type="FuseMSDeformAttention",
                        # type="MultiScaleDeformableAttention",
                        embed_dims=256,
                        num_levels=5,
                        dropout=0.0,
                    ),
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=("self_attn", "norm", "ffn", "norm"),
                ),
            ),
            decoder=dict(
                type="DinoTransformerDecoder",
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type="DetrTransformerDecoderLayer",
                    attn_cfgs=[
                        dict(
                            type="MultiheadAttention",
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.0,
                        ),
                        dict(
                            type="MultiScaleDeformableAttention",
                            embed_dims=256,
                            num_levels=5,
                            dropout=0.0,
                        ),
                    ],
                    feedforward_channels=2048,
                    ffn_dropout=0.0,
                    operation_order=(
                        "self_attn",
                        "norm",
                        "cross_attn",
                        "norm",
                        "ffn",
                        "norm",
                    ),
                ),
            ),
        ),
        positional_encoding=dict(
            type="SinePositionalEncoding", num_feats=128, temperature=20, normalize=True
        ),
        loss_cls=dict(  # Different from the DINO
            type="QualityFocalLoss", use_sigmoid=True, beta=2.0, loss_weight=1.0
        ),
        loss_bbox=dict(type="L1Loss", loss_weight=5.0),
        loss_iou=dict(type="GIoULoss", loss_weight=2.0),    # TODO -> DIoU?
    ),
    rpn_head=dict(
        type="RPNHead",
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type="AnchorGenerator",
            octave_base_scale=4,
            scales_per_octave=3,
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64, 128],
        ),
        bbox_coder=dict(
            type="DeltaXYWHBBoxCoder",
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0],
        ),
        loss_cls=dict(
            type="CrossEntropyLoss",
            use_sigmoid=True,
            loss_weight=1.0 * num_dec_layer * loss_lambda,
        ),  # 影响 loss_rpn_cls 一项
        loss_bbox=dict(type="L1Loss", loss_weight=1.0 * num_dec_layer * loss_lambda),   # 影响 loss_rpn_bbox 一项 [可改成GIOU等]
    ),
    roi_head=[
        dict(
            type="CoStandardRoIHead",
            bbox_roi_extractor=dict(
                type="SingleRoIExtractor",
                roi_layer=dict(type="RoIAlign", output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56,
            ),
            bbox_head=dict(
                type="Shared2FCBBoxHead",
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type="DeltaXYWHBBoxCoder",
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2],
                ),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type="CrossEntropyLoss",
                    use_sigmoid=False,
                    loss_weight=1.0 * num_dec_layer * loss_lambda,
                ),  # TODO -> Focal?
                loss_bbox=dict(
                    type="GIoULoss", loss_weight=10.0 * num_dec_layer * loss_lambda
                ),  # TODO -> DIoU?
            ),
        )
    ],
    bbox_head=[
        dict(
            type="CoATSSHead",
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type="AnchorGenerator",
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128],
            ),
            bbox_coder=dict(
                type="DeltaXYWHBBoxCoder",
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2],
            ),
            loss_cls=dict(
                type="FocalLoss",
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda,
            ),
            loss_bbox=dict(
                type="GIoULoss", loss_weight=2.0 * num_dec_layer * loss_lambda
            ),  # TODO -> DIoU?
            loss_centerness=dict(
                type="CrossEntropyLoss",
                use_sigmoid=True,
                loss_weight=1.0 * num_dec_layer * loss_lambda,
            ),
        ),
    ],
    # model training and testing settings
    train_cfg=[
        dict(
            assigner=dict(
                type="HungarianAssigner",
                match_costs=[
                    dict(type="FocalLossCost", weight=2.0),
                    dict(type="BBoxL1Cost", weight=5.0, box_format="xywh"),
                    dict(type="IoUCost", iou_mode="giou", weight=2.0),
                ],
            )
        ),
        dict(
            rpn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.3,
                    min_pos_iou=0.3,
                    match_low_quality=True,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=256,
                    pos_fraction=0.5,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=False,
                ),
                allowed_border=-1,
                pos_weight=-1,
                debug=False,
            ),
            rpn_proposal=dict(
                nms_pre=4000,
                max_per_img=1000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=dict(
                assigner=dict(
                    type="MaxIoUAssigner",
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1,
                ),
                sampler=dict(
                    type="RandomSampler",
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True,
                ),
                pos_weight=-1,
                debug=False,
            ),
        ),
        dict(
            assigner=dict(type="ATSSAssigner", topk=9),
            allowed_border=-1,
            pos_weight=-1,
            debug=False,
        ),
    ],
    test_cfg=[
        # Deferent from the DINO, we use the NMS.
        dict(
            max_per_img=300,
            # NMS can improve the mAP by 0.2.
            nms=dict(type="soft_nms", iou_threshold=0.8),
        ),
        dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type="nms", iou_threshold=0.7),
                min_bbox_size=0,
            ),
            rcnn=dict(
                score_thr=0.0, nms=dict(type="nms", iou_threshold=0.5), max_per_img=100
            ),
        ),
        dict(
            # atss bbox head:
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type="nms", iou_threshold=0.6),
            max_per_img=100,
        ),
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ],
)

# train_pipeline = [
#     dict(
#         type='MultiInputMosaic',
#         keys=['img', 'tir'],
#         prob=1.0,
#         img_scale = (640, 512),
#         center_ratio_range = (0.7, 1.3), 
#         pad_val = 114.0,
#         bbox_clip_border=False,
#         individual_pipeline=[
#             dict(type="LoadImageFromFile"),
#             dict(type="LoadTirFromPath"),
#             dict(type="LoadAnnotations", with_bbox=True),
#             dict(
#                 type="BugFreeTransformBroadcaster",
#                 mapping={
#                     "img": ["tir", "img"],
#                     "img_shape": ["img_shape", "img_shape"],
#                     "gt_bboxes": ['gt_bboxes', 'gt_bboxes'],
#                     "gt_bboxes_labels": ['gt_bboxes_labels', 'gt_bboxes_labels'],
#                     "gt_ignore_flags": ['gt_ignore_flags', 'gt_ignore_flags'],
#                 },
#                 auto_remap=True,
#                 share_random_params=True,
#                 transforms=[
#                     dict(
#                         type='RandomCrop',
#                         crop_type='absolute_range',
#                         crop_size=(576, 640),
#                         allow_negative_crop=False,
#                     ),
#                     dict(type='RandomFlip', prob=0.5),
#                     dict(type='Resize', scale=(640, 512)),
#                 ],
#             ),
#         ]
#     ),
#     # dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
#     dict(type="CustomPackDetInputs"),
# ]

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadTirFromPath"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="BugFreeTransformBroadcaster",
        mapping={
            "img": ["tir", "img"],
            "img_shape": ["img_shape", "img_shape"],
            "gt_bboxes": ['gt_bboxes', 'gt_bboxes'],
            "gt_bboxes_labels": ['gt_bboxes_labels', 'gt_bboxes_labels'],
            "gt_ignore_flags": ['gt_ignore_flags', 'gt_ignore_flags'],
        },
        auto_remap=True,
        share_random_params=True,
        transforms=[
            dict(
                type='RandomCrop',
                crop_type='absolute_range',
                crop_size=(576, 640),
                allow_negative_crop=False,
            ),
            dict(type='RandomFlip', prob=0.5),
            dict(type='Resize', scale=(640, 512)),
        ],
    ),
    dict(type='RandomShiftOnlyImg', max_shift_px=10, prob=0.5),
    dict(type='AdaptiveHistEQU'),
    dict(type="CustomPackDetInputs"),
]

train_dataloader = dict(
    batch_size=3,
    num_workers=16,
    prefetch_factor=4,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        serialize_data=False,
        ann_file="annotations/train.json",
        data_prefix=dict(img_path="train/rgb", tir_path="train/tir"),
        # filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
    ),
)

val_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadTirFromPath"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="BugFreeTransformBroadcaster",
        mapping={
            "img": ["tir", "img"],
            "img_shape": ["img_shape", "img_shape"],
            "gt_bboxes": ['gt_bboxes', 'gt_bboxes'],
            "gt_bboxes_labels": ['gt_bboxes_labels', 'gt_bboxes_labels'],
            "gt_ignore_flags": ['gt_ignore_flags', 'gt_ignore_flags'],
            "scale_factor": ['scale_factor', 'scale_factor'],
        },
        auto_remap=True,
        share_random_params=True,
        transforms=[
            dict(type='Resize', scale=(640, 512)),
        ],
    ),
    # dict(type='Resize', scale=(640, 512)),
    #dict(type='Resize', scale_factor=1.0),
    # dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),

    dict(type='AdaptiveHistEQU'),
    # dict(type='RandomShiftOnlyImg', max_shift_px=10, prob=0.5),

    dict(
        type="CustomPackDetInputs",
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')  # as is. keep `scale_factor`
    ),
]

val_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        serialize_data=False,
        data_root=data_root,
        ann_file="annotations/val.json",
        data_prefix=dict(img_path="val/rgb", tir_path="val/tir"),
        # filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=val_pipeline,
    ),
)
val_evaluator = dict(
    type="CocoMetric",
    ann_file=f"{data_root}/annotations/val.json",
    metric="bbox",
    format_only=False,
    backend_args=backend_args,
)

# test_dataloader = val_dataloader
# test_evaluator = val_evaluator

test_dataloader = dict(
    batch_size=1,
    num_workers=16,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        serialize_data=False,
        data_root=data_root,
        ann_file="annotations/test.json",
        data_prefix=dict(img_path="test/rgb", tir_path="test/tir"),
        # filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=val_pipeline,
    ),
)

test_evaluator = dict(
    type="CocoMetric",
    ann_file=f"{data_root}/annotations/test.json",
    metric="bbox",
    format_only=False,
    backend_args=backend_args,
)

base_lr = 2e-5
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=base_lr, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        "query_head.cls_branches": dict(lr_mult=10),
        "roi_head.0.bbox_head.fc_cls": dict(lr_mult=10),
        "roi_head.0.bbox_head.fc_reg": dict(lr_mult=10),
        "bbox_head.0.atss_cls": dict(lr_mult=10),
    }),
)

max_epochs = 14
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1./4,
        by_epoch=False,
        begin=0,
        end=1000,
        # verbose=True
    ),
    dict(
        type='CosineAnnealingLR',
        eta_min=base_lr * 0.02,
        begin=1,
        end=16,
        T_max=16,
        by_epoch=True,
        verbose=True,
        # convert_to_iter_based=True,
    ),
]

log_level = "INFO"
log_processor = dict(by_epoch=True)

# NOTE: `auto_scale_lr` is for automatically scaling LR,
# USER SHOULD NOT CHANGE ITS VALUES.
# base_batch_size = (8 GPUs) x (2 samples per GPU)
auto_scale_lr = dict(base_batch_size=6)
