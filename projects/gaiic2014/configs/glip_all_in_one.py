default_scope = 'mmdet'

custom_imports = dict(
    imports=[
        "projects.CO-DETR.codetr",
        "projects.gaiic2014.core",
    ],
    allow_failed_imports=False,
)


default_hooks = dict(
    checkpoint=dict(interval=1, type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))

env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
backend_args = None
resume = False

data_root = "data/track1-A/"
dataset_type = 'GAIIC2014DatasetV2'
lang_model_name = 'bert-base-uncased'
num_classes = 5


load_from = 'ckpt/glip_atss_swin-l_fpn_dyhead_16xb2_ms-2x_funtune_coco_20230910_100800-e9be4274.pth'
model = dict(
    type='GLIP',
    backbone=dict(
        type='SwinTransformer',
        attn_drop_rate=0.0,
        convert_weights=False,
        depths=[2, 2, 18, 2],
        drop_path_rate=0.4,
        drop_rate=0.0,
        embed_dims=192,
        mlp_ratio=4,
        num_heads=[6, 12, 24, 48],
        out_indices=(1, 2, 3),
        patch_norm=True,
        qk_scale=None,
        qkv_bias=True,
        window_size=12,
        with_cp=False
    ),
    bbox_head=dict(
        anchor_generator=dict(
            center_offset=0.5,
            octave_base_scale=8,
            ratios=[1.0],
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
            type='AnchorGenerator'
        ),
        bbox_coder=dict(
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[0.1, 0.1, 0.2, 0.2],
            type='DeltaXYWHBBoxCoderForGLIP'
        ),
        early_fuse=True,
        feat_channels=256,
        in_channels=256,
        lang_model_name=lang_model_name,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_centerness=dict(loss_weight=1.0, type='CrossEntropyLoss', use_sigmoid=True),
        loss_cls=dict(
            alpha=0.25,
            gamma=2.0,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True
        ),
        num_classes=num_classes,
        num_dyhead_blocks=8,
        type='ATSSVLFusionHead',
        use_checkpoint=True
    ),
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        bgr_to_rgb=False,
        mean=[103.53, 116.28, 123.675],
        pad_size_divisor=32,
        std=[57.375, 57.12, 58.395],
    ),
    language_model=dict(name=lang_model_name, type='BertModel'),
    neck=dict(
        type='FPN_DropBlock',
        add_extra_convs='on_output',
        in_channels=[384, 768, 1536],
        num_outs=5,
        out_channels=256,
        relu_before_extra_convs=True,
        start_level=0,
    ),
    test_cfg=dict(
        max_per_img=300,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.8, type='soft_nms'),
        nms_pre=1000,
        score_thr=0.05
    ),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            iou_calculator=dict(type='BboxOverlaps2D_GLIP'),
            topk=9,
            type='ATSSAssigner'
        ),
        debug=False,
        pos_weight=-1),
)


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, imdecode_backend='pillow'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomShiftOnlyImg', max_shift_px=10, prob=0.5),
    dict(type="LoadTirFromPath"),
    dict(type="ImgBlend", fuse_keys=['img', 'tir'], alphas=[0.25, 0.75]),
    dict(type='GTBoxSubOne_GLIP'),
    dict(
        type='RandomChoiceResize',
        backend='pillow',
        keep_ratio=True,
        resize_type='FixScaleResize',
        scales=[(1333, 480), (1333, 560), (1333, 640), (1333, 720), (1333, 800)],
    ),
    dict(type='RandomFlip_GLIP', prob=0.5),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'flip', 'flip_direction', 'text', 'custom_entities'),
    ),
]

train_dataloader = dict(
    batch_size=2,
    num_workers=16,
    prefetch_factor=4,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        serialize_data=False,
        ann_file="annotations/train.json",
        data_prefix=dict(img_path="train/rgb", tir_path="train/tir"),
        pipeline=train_pipeline,
        return_classes=True,
    ),
)

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args, imdecode_backend='pillow'),
    dict(type='FixScaleResize', backend='pillow', keep_ratio=True, scale=(640, 512)),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type="LoadTirFromPath"),
    dict(type="ImgBlend", fuse_keys=['img', 'tir'], alphas=[0.25, 0.75]),
    dict(
        type='PackDetInputs',
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
            'text',
            'custom_entities',
        )
    ),
]


val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        type=dataset_type,
        serialize_data=False,
        data_root=data_root,
        ann_file="annotations/val.json",
        data_prefix=dict(img_path="val/rgb", tir_path="val/tir"),
        pipeline=val_pipeline,
        return_classes=True,
    ),
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler')
)
val_evaluator = dict(
    type='CocoMetric',
    metric='bbox',
    ann_file=f"{data_root}/annotations/val.json",
    format_only=False,
    backend_args=backend_args,
)

test_dataloader = dict(
    batch_size=1,
    drop_last=False,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'),
    dataset=dict(
        type=dataset_type,
        serialize_data=False,
        data_root=data_root,
        ann_file="annotations/test.json",
        data_prefix=dict(img_path="test/rgb", tir_path="test/tir"),
        pipeline=val_pipeline,
        return_classes=True,
    ),
)
test_evaluator = dict(
    type="CocoMetric",
    ann_file=f"{data_root}/annotations/test.json",
    metric="bbox",
    format_only=False,
    backend_args=backend_args,
)


optim_wrapper = dict(
    type='OptimWrapper',
    clip_grad=dict(max_norm=1, norm_type=2),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=1e-05, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0)
        )
    ),
)

max_epochs = 10
train_cfg = dict(max_epochs=max_epochs, type='EpochBasedTrainLoop', val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(type='LinearLR', begin=0, by_epoch=False, end=500, start_factor=0.001),
    dict(
        type='MultiStepLR',
        begin=0,
        by_epoch=True,
        end=max_epochs,
        gamma=0.1,
        milestones=[5, 8],
    ),
]

log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
auto_scale_lr = dict(base_batch_size=16, enable=False)
