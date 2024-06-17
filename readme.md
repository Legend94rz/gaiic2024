# 代码说明

## 训练配置

显卡：A100 40G x 2
内存: 100G。实际使用可能40G左右。
batch_size=3, backbone.with_cp=False，transformer.encoder.with_cp=True，显存开销~38G，每个模型大约需要1.5天。

## 环境配置
项目依赖项并不多，可以直接看`init.sh`

实际使用的一些关键包的版本如下：
```
mmcv                      2.1.0                    pypi_0    pypi
mmdet                     3.3.0                    pypi_0    pypi
mmengine                  0.10.3                   pypi_0    pypi
pytorch                   2.2.2           py3.10_cuda11.8_cudnn8.7.0_0    pytorch
```
版本号并不一定需要完全一致，仅供参考.

## 数据
> 在提交B榜时，额外集成使用外部数据训练的两个模型并没有带来十分显著的收益(不会导致名次差异)，而因为这部分代码实现比较仓促，大部分路径都在代码里写死了，不好适配复现环境。如果不需要严格复现，可以不处理外部数据及相关模型，同时可以跳过此节。`train.sh`默认已经把相关代码注释掉了，如确实有需要，可以手动取消注释。

使用两个公开数据：
* aistudio: https://aistudio.baidu.com/datasetdetail/206266/0
* vedai: https://downloads.greyc.fr/vedai/

其中aistudio**需要手动下载**，vedai的下载方式见`train.sh`。

数据需要解压分别放在 `data/aistudio` 与 `data/vedai` 目录下，最终`data`目录树如下所示：
```
data
├── [other folders]
├── aistudio
│   ├── original
│   |   └── original
│   |       ├── annotations
│   |       └── imgs
|   ├── data.csv
|   └── roundabouts.csv
└── vedai
    ├── Annotations512
    └── Vehicules512
```

然后直接`python tools/external_data.py`会把这两个外部数据都处理成COCO的标注格式备用。

数据的具体使用详见后文。


## 预训练模型
使用 CoDETR-SwinL-16Epoch-DETR-o365+COCO 的预训练权重，即 [这个表格](https://github.com/open-mmlab/mmdetection/blob/v3.2.0/projects/CO-DETR/README.md#results-and-models) 里的最后一行。

另外建议同时下载 swin-large backbone的ckpt，否则需要注释配置里的`pretrained`字段。

```
wget https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth

wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
```

## 算法

### 整体思路介绍
最终b榜提交两次:

9模型: 分数 0.48887175566920904 (包含两个外部数据模型)

3模型: 分数 0.48128947817837825 (无需外部数据)

-------

所有模型都基于前述 Co-DETR，区别在于训练数据、pipeline、网络结构、加载的预训练参数不同:
* 对于训练数据，我们手动清洗了训练集和验证集，修正了大部分的类别标注，bbox基本不做修正。
* 对于数据增强pipeline，大致有两种：一种是常规的一阶段数据增强pipeline；一种是两阶段的，即前x个epoch先用一些强增广，后x个epoch切换为常规增广pipeline。
* 对于网络结构方面，为了最大程度地利用上预训练的参数，仅对`query_head.transformer.encoder`中的attention做了两种变形，以融合双光。
* 对于预训练参数，也有两种：一种是直接加载官方权重（下载方式见前文）；一种是基于官方权重，额外使用外部数据训练，最后在比赛数据上训练模型。

### 数据

我们用一些标注工具，对训练集、验证集重新人工清洗，主要修改 truck/van/freight_car 三类的类别标注，最终保留了两个版本的（训练）标注文件：
* version 0518: `data/track1-A/annotations/train_0518.json`
* version 0527: `data/track1-A/annotations/train_0527.json`

两者差异主要在van类别，0527清洗得更加激进：把一些原来官方标注为car的也调整为van。

验证集只有一个版本 `data/track1-A/annotations/val_0527.json`

### 数据增强 pipeline

* 常规一阶段增强
    ```
    train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadTirFromPath"),
        dict(type="LoadAnnotations", with_bbox=True),
        dict(type='AdaptiveHistEQU'),       # 自适应对比度
        dict(type='RandomShiftOnlyImg', max_shift_px=10, prob=0.5), # 随机平移RGB图像
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
                    type='RandomResize',
                    scale=image_size,
                    ratio_range=(0.75, 1.5),
                    keep_ratio=True
                ),
                dict(
                    type='RandomCrop',
                    crop_type='absolute',
                    crop_size=image_size,
                    allow_negative_crop=False,
                ),
                dict(type='RandomFlip', prob=0.5),
                dict(type='Pad', size=image_size, pad_val=dict(img=(114, 114, 114))),
            ],
        ),
        dict(type="CustomPackDetInputs"),
    ]
    ```

* 两阶段增强pipeline

    先采用带 mosaic(或类似) 的增广方式训练几个epoch，然后切换为上述常规pipeline。因为mosaic增广的图像更加丰富，但小目标经常会在边界处截断，反而不利于模型学习，完全采用mosaic的增广pipeline会导致模型性能下降。

    带mosaic的pipeline大致如下（**不同模型略有差异**，详见配置代码）：
    ```
    train_pipeline_stage1 = [
        dict(
            type='MultiInputMosaic',
            keys=['img', 'tir'],
            prob=1.0,
            img_scale=image_size,
            center_ratio_range=(0.5, 1.05), 
            pad_val=114.0,
            bbox_clip_border=True,
            individual_pipeline=[
                dict(type="LoadImageFromFile"),
                dict(type="LoadTirFromPath"),
                dict(type="LoadAnnotations", with_bbox=True),
                dict(type='AdaptiveHistEQU'),
                dict(type='RandomShiftOnlyImg', max_shift_px=10, prob=0.5),
                dict(**transform_broadcast, transforms=[
                dict(type='RandomRotate90', prob=0.5, dir=['l']),
                dict(type='RandomFlip', prob=0.5, direction=['horizontal', 'vertical', 'diagonal']),
                ])
            ]
        ),
        dict(
            **transform_broadcast,
            transforms=[
                dict(
                    type='RandomAffine',
                    scaling_ratio_range=(0.65, 1.5),
                    max_translate_ratio=0.1,
                    max_rotate_degree=0,
                    max_shear_degree=0,
                    border=(-image_size[0] // 2, -image_size[1] // 2)
                ),
                dict(
                    type='RandomResize',
                    scale=image_size,
                    ratio_range=(0.8, 1.1),
                    keep_ratio=True
                ),
                dict(type='Pad', size_divisor=4, pad_val=dict(img=(114, 114, 114))),
            ],
        ),
        dict(type='FilterAnnotations', min_gt_bbox_wh=(3, 3), keep_empty=False),
        # dict(type='Normalize', mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True),
        dict(type="CustomPackDetInputs"),
    ]
    ```
    训练约7个epoch后会切换为常规pipeline。切换的操作可以通过`PipelineSwitchHook`实现，无需手动。



### 网络结构
原始的 Co-DETR 仅支持RGB图像检测，为了支持输入两个模态的图像，同时尽量不对网络结构有较大改动，从而能利用上尽可能多的预训练参数，两种改动只针对 Encoder 的 Attention 部分。

具体有以下两种变式：

* mean

    Encoder每一层计算self-attention时，query除了与自身计算attention（RGB特征的self-attention），还与红外光特征计算一次attention，然后两者取平均作为输出。

* concat

    Encoder每一层计算self-attention时，RGB特征与红外特征直接在长度维concat作为query，双光的特征图也直接concat，原来每个模态默认有4个不同尺度的特征图，修改后共有8个特征图作为value。此时query（及输出）的特征数量变为原来两倍，最终Encoder输出时，只取前一半query对应的输出即可。
由于特征图数量有改动，需要相应修改 DeformAttn 中 reference point 的参数形状，`scripts/patch_ckpt.py`完成此功能。



### 预训练参数

存在两种预训练参数，用于模型初始化：

* official

    直接使用官方权重

* pretrain

    先加载官方权重，然后合并使用前述两个外部数据训练一个模型，训练完成后，其他模型会加载该模型参数继续在官方数据上训练。



### 模型集成

通过选用不同的组件，可以搭配出不同的模型。B榜实际提交两次：

* 9模型版本

|   | 数据增强pipeline   | 网络结构   | 数据 | 预训练参数 | 备注 | 文件名 |
|---|----------------------|--------|------|-----------|--------|--------|
| 1 | 一阶段               | concat | 0518 | official  |        |codetr_full_0518data|
| 2 | 两阶段               | concat | 0527 | official  |        |codetr_full_0527data|
| 3 | 两阶段               | mean   | 0527 | official  |        |mean_fuse_full|
| 4 | 两阶段               | mean   | 0527 | pretrain  |        |mean_fuse_with_pretrained|
| 5 | 两阶段               | concat | 0527 | official  | fold 0 |codetr_0527fold0|
| 6 | 两阶段               | concat | 0527 | official  | fold 1 |codetr_0527fold1|
| 7 | 两阶段               | concat | 0527 | official  | fold 2 |codetr_0527fold2|
| 8 | 两阶段(更强)         | concat | 0527 | official  |        |codetr_full_0527data_strong_aug|
| 9 | 两阶段(更强)         | concat | 0527 | pretrain  |        |codetr_full_0527data_strong_aug_with_pretrain|

ensemble 参数: skip/nms = 0.07/0.75

A榜	0.5384122069865042，B榜 0.48887175566920904

其中模型5/6/7是用模型2的数据划分5折后的前三折数据分别训练得到，5折划分训练数据由脚本`scripts/split_nfold.py`提供。

* 3模型版本

|   | 数据增强pipeline   | 网络结构   | 数据 | 预训练参数 | 备注 | 文件名 |
|---|----------------------|--------|------|-----------|--------|--------|
| 2 | 两阶段               | concat | 0527 | official  |        |codetr_full_0527data|
| 3 | 两阶段               | mean   | 0527 | official  |        |mean_fuse_full|
| 5 | 两阶段               | concat | 0527 | official  | fold 0 |codetr_0527fold0|

ensemble 参数: skip/nms = 0.05/0.7

A 榜 0.5316742459794518，B 榜 0.48128947817837825

### 算法的其他细节

* SWA
* lr schedular
* 从A榜来看，10个epoch分数最高，一般不取最后一个epoch的权重
* 推理时超参: soft_nms `iou_thres=0.7`

## 训练流程
见`train.sh`

`bash train.sh`

## 测试流程
见`test.sh`， 运行时**需要指定参数**:
`bash test.sh [input_dir] [data_root] [output_json]`

`input_dir`放测试数据，`data_root`是`input_dir`的上级目录，`output_json`是输出文件路径（目前需要保证其目录已经存在）。

例如：

`bash test.sh data/track1-A/test data/track1-A  data/result/pred.json`

具体也可以参考`index.py`里的调用方法。

## 其他注意事项
* `train.sh` 没有完全测试，可能不一定跑通。
* 为简化复现训练流程，可以忽略采用外部数据的两个模型，仅集成其余7个模型不影响线上成绩；仅复现B榜提交的3个模型，也不影响B榜排名，与A榜约有6个千分点的差异，可能影响A榜排名。
* 训练时采用`batch_size=3`， 约需40G显存，A100 x 2 每个模型需要约 1.5 天。复现时如果显存不够可以考虑把`backbone`的`with_cp`设为`True`，或者`batch_size=1`，但训练时间会更长。
* 代码组织结构与规范略有不同。`projects`里是核心代码；`scripts`目录里是一些自定义工具类脚本，例如5折划分、生成提交文件、可视化等等；`tools`里是来自`mmdet`里的`tools`文件夹，主要是`train.py`与`test.py`,对比官方略有修改。下载的预训练权重会放在`ckpt`文件夹。
* 由于训练使用的是清洗过的标注，请确保使用`data/track1-A/annotations`文件夹里的标注文件来复现训练。
