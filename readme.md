[x] 线上线下差距较大的问题(是否与score阈值有关? maxDet?): 不要加阈值过滤。仍然没有完全解决：需要确认下是否是val与test分布差距较大，看下预测结果。
    大概率是因为测试集都没对齐。另外有一些极小的目标，暂时无解。
    重点解决【有一些明明rgb上很清晰，但是就是漏检】: 随机用其他无目标区域填充tir
[x] 数据可视化；预测结果 badcase
[x] 排查训练速度问题
[x] test json
[x] 检查下各种resize/crop参数的正确性，w/h的顺序
[x] 检查下各种bbox的格式：到底是min/max还是min/wh还是cxy/wh: train/val/test pkl/output json.
    标注与提交都是min/wh
[x] 验证单模态(tir/rgb)的分数 (as baseline)
[] 验证mosaic对于CoDETR的有效性
[x] appearance augmentation || fuse augmentaion
[x] wandb。尽快确定改动是否有效
[] giouloss -> ciou loss / siou loss
[x] lr warmup & annealing
[x] swa
[x] 测试下 test_cfg 的nms参数对测试集上大量目标的图片是否不够。【似乎足够，测试nq=300, nms-thres->0.9，分数降低】
[x] 检查改进的concat fuse deform attn的几个重要参数的正确性（不同模态的参数是否变化不同）：有效
[] 检查 multi-scale是否有助于解决漏检的问题（若有，则需要加 **TTA**，并且调整multi-scale增广）
[x] 检查是否有类似 p_obj 的输出，调节权重：FocalLoss use_sigmoid，不匹配的query预测是全0，而不是one-hot。
[x] *验证集加入训练*
[] 输出后处理：如先确定recall接近1时的conf-thres, 在这附近，找一个conf-下降最快的点，低于该点的框置0（或者不输出）. 另外可能有面积极小甚至为0的框，需要去掉。
[] weighted box fusion。官方实现很可能有bug，需要自己写
[] soft-label?
[x] 把那几个过小的，手动放大，测试下有无改善。若有，针对这部分数据特殊处理。【无改善】
[] 确认下，good/bad分别是什么原因导致的，如果bad里预测没问题，而仅仅是标注错了，那数据清洗的收益可能不大。以及good是否是真good，还是因为预测和标注都错了所以分数高。
[] 数据清洗(规则 + ignore<有无可能做在线?>) 对于训练+验证集：过大的van(>150): 规则筛选后修改标注。(100~150?): 在线ignore
[] 训练集目前发现的问题: 
    van与freight_car类别相互误标；
    van里还误标了一些car
    truck/bus标注相对干净，只有少量car/van被误标。
[] 用清洗过数据的模型重新测试 nq/nms/type(soft_nms|nms) 参数：先在验证集上大致验证再提交。
[] fiftyone 检查下test集预测结果，到底是什么原因导致线上分数很低？: 
[x] 去掉 lr cos annealing 试试
[] scale 0.8~1.2
[] randomrotate 90(需要手写，mmdet里的不支持broadcast)
[] mean fuse 再调下pipeline，看看上限
[] ensemble / sahi

**WARN**
由于B阶段只有一天时间，两次提交机会。每次提交之后一定注意要保存最优的模型权重，一旦丢失，当天是来不及重新训练的。
在比赛过程中注意保存最优的两个权重。同时A阶段结束之前3天(6月7号)再次检查是否缺失。

**WARN**
2024年5月17日8:00之前发邮件到大赛邮箱data@tsinghua.edu.cn
使用的模型权重（下载链接、md5）及数据集


**时间节点**
0608 10:00 开始准备复现环境
0610 23:00 线下提交结束、复现环境搭建结束
0611 10:00 ~ 23:00 B榜
WANDB 注意关掉


速度问题:
注意提交时不要 加--show参数，很慢。
with mosaic:
    backbone with_cp=False, encoder with_cp=6, batchsize=1: OOM (eta: >4 days)
    backbone with_cp=True, encoder with_cp=4,  batchsize=2: OOM (eta: >4 days)

without mosaic:
    [only tir] backbone with_cp=False, encoder with_cp=6, batchsize=3: ? (eta: ~2 days)
    [2 modalities] backbone with_cp=False, encoder with_cp=6, batchsize=3 (eta ~2d 8h)


更小的img size
with_cp
smaller layer / 400 query: 可能会显著影响指标
larger batch_size: 需要改代码。主要是由于mosaic带来的影响，每个batch里有些样本有mosaic，有些没有，导致分辨率不同。简单的改法是mosaic==1.0

#### submit records

conf-thres=0.5, mosaic + 多模态: 1epoch     线上 0.2686912743221043
conf-thres=0.5, 单模态: 1 epoch             线上 0.22698282977757558

conf-thres=0    单模态: 1 epoch             线上 0.38238603322554376
conf-thres=0    单模态: 10 epoch            线上 0.38908813383119695    (0.624)
conf-thres=0    多模态: 1 epoch             线上 0.39385607852916976    (0.514)
conf-thres=0    多模态: 10 epoch            线上 0.4477186164130829     (0.649)
conf-thres=0    多模态: 16 epoch            线上 0.43890585216012845    (0.652): 多训练可能并不会有线上收益! 还需要测试

conf-thres=0    多模态+lrsch+shift+autocontrast: 9 epoch                线上 0.4467967757882764 (0.638)
conf-thres=0    多模态+lrsch+shift+autocontrast: 10 epoch               线上 0.4480350955482753 (0.638)

conf-thres=0    多模态+lrsch+shift+adaptiveHistEQU+Copypaste+randomsmear: 10 epoch               线上 0.43570007604618116  (0.616) 【更复杂的增广可能没收益了】
conf-thres=0    多模态+lrsch+shift+adaptiveHistEQU+Copypaste+randomsmear: 16 epoch               线上 0.4282321671406016   (0.632)  【16个epoch 已经过拟合】
SWA / loss function / 怎么解决漏检过多的问题？

conf-thres=0    多模态+lrsch+shift+autocontrast: 16 epoch               线上 0.4361348096499168 (0.650)  这种融合方式的最优epoch在[10, 15]之间
                多模态+lrsch+shift+AdaptiveHistEQU+swa+concat deform attn: 10epoch max14e  线上 0.45847593272744164 (0.649) 
                多模态+lrsch+shift+AdaptiveHistEQU+swa+concat deform attn: 14epoch max14e  线上 0.45142205638384225 (0.650) 

_20240428_173223:
    **多模态, const lr + cos annealing, 10/13epoch, randshift+AdaptiveHistEQU+swa+concat deform attn: 线上 0.4901899498048514 (0.644 0.866 0.769 0.322 0.647 0.831)** tta 0.4683023883758988 可能参数没调好，需要检查下结果，并且tta可能超时(1h) truck 线上: 0.454919137353518
    多模态, const lr + cos annealing, 13/13epoch, randshift+AdaptiveHistEQU+swa+concat deform attn: 线上 0.4892719064102989 (0.647 0.868 0.770 0.324 0.650 0.835)

_20240502_114151
    基于_20240428_173223，调大QualityFocalLoss权重，SWA调早1个epoch，增大MultiScale/Shift范围，pad->0， 10/12 epoch 线上 0.4897002480529805 (0.641 0.863 0.769 0.322 0.644 0.839)
    ... 12/12 epoch 线上 0.48846226714393864 (0.643)

_20240503_215119
    基于 _20240502_114151，SWA 调回0428。 9/13 epoch 线上 0.4845854188056091 (0.643) 理论上这个是单模型. [ 有机会测试下7/8epoch的 ]
    ...  10/13 epoch 线上 0.4861246502594757 (0.644)
    ...  11/13 epoch 线上 0.48600053907053986 (0.646)  
    ... 8/13 epoch 线上 0.47316129186568373 (0.639) :    【单模差不多是9/10个epoch最高】

_20240505_140046
    基于 _20240502_114151，回调QualityFocalLoss权重。 10/12 epoch 线上 0.4882847270360551 (0.642 0.860 0.770 0.277 0.645 0.839)
    基于 _20240502_114151，回调QualityFocalLoss权重。 12/12 epoch 线上 0.4858214711495492 (0.644 0.862 0.770 0.287 0.646 0.840)

20240507: 基于 _20240428_173223 10 epoch, 去掉边缘小于5像素的框: 0.490305915043059. 提高可以忽略不计。

_20240507_105236:
    基于 _20240428_173223 训练+验证集合并训练。 10/13 epoch 线上 0.48946345511761813 (0.706 0.926 0.841 0.376 0.712 0.874 <验证集结果仅作参考>)

_20240511_153021:
    大致基于 _20240428_173223, 修改RandomCrop，Resize离散化、涂黑Crop边缘过小的目标并删除box（想解决图像边缘生成过多无效box的问题）。10/13 epoch 线上 0.4841715521016622 (0.642 0.859 0.767 0.318 0.646 0.835)
    ... 11/13 epoch 线上 0.4848613478041527 (0.644 0.861 0.772 0.313 0.648 0.837) 无收益

_20240512_180416
    配置基于 _20240428_173223
    训练集数据清洗。这个版本验证集暂时没清洗，线下分数无意义。只看线上是否有收益。
    训练集清洗 1~5100 + 15472-17990 (train_updated.json)
    10/13 epoch 线上 0.4989366097228196
    **12/13 epoch 线上 0.5005958718437866 (修正后验证集 0.602 0.810 0.719 0.301 0.611 0.643)**
    13/13 epoch 线上 0.4994995678118088 (修正后验证集 0.602 0.812 0.721 0.303 0.612 0.643 还是稍微有些过拟合。验证集并不完全干净)

_20240514_173554
    基于 _20240512_180416
    加 DualModalCutOut
    10/13 线上 0.4979335950223399 (0.600 0.809 0.720 0.312 0.611 0.658)
    12/13 线上 0.498425164277342  (0.602 0.811 0.720 0.316 0.612 0.660)
        子类别线上分数
        van(0.05716354848148401*5=0.28581774240742003),
        bus(0.13601968979420329*5=0.6800984489710165)

_20240516_094744
    基于 _20240428_173223。*没有 DualModalCutOut*
    基本完全清洗数据。
    **10/13 线上 0.3713848057814872 + 0.13601194238769948(bus) = 0.5073967481691867 (0.636 0.857 0.762 0.326 0.643 0.666)**
    12/13 线上 0.36962252792366507+ 0.13644064670397224(bus) = 0.506063174627637  (0.639 0.860 0.764 0.335 0.645 0.670)
    需要检查下 test检测结果。可能是van过拟合了。


_20240517_170556
    基于 _20240516_094744
    调整了 lr_sch 最终的学习率: eta_min=base_lr * 0.5
    10/13 epoch 线上 0.5020464560493408 (0.641 0.861 0.764 0.304 0.648 0.680)
    12/13 epoch 线上 0.5019061583099873 (0.644 0.864 0.767 0.318 0.651 0.682)

_20240518_234754
    基于 _20240428_173223
    调整训练集中前 5000 张的van标注: 原来大部分类似van但标成car的
    *验证集没有同步修改，因此可能不具有参考意义*
    改了 test soft_nms 阈值 0.8->0.7
    **epoch 9/13 线上 0.44252119400472606 + 0.07489918001367844(van) = 0.5174203740184045 (0.616 0.829 0.736 0.306 0.621 0.694)**
        truck: 0.09277150236166*5 = 0.4638575118083
        freight_car: 0.10356438085956061*5 = 0.517821904297803
        car(大致): 0.109 * 5 = 0.545
        |0.545      0.4638575118083     0.6800597119384975     0.37449590006839223     0.517821904297803|

    epoch 10/13 线上 0.4428001436718439 + 0.07365442512415667(van) = 0.5164545687960006 (0.617 0.829 0.736 0.312 0.620 0.686)


_20240520_153439
    基于 _20240518_234754。*验证集同样是_20240518_234754，不太具有参考意义。*
    Sparse4D Denoise Query
    10/13 epoch 线上 0.5188770992686201 (0.618 0.830 0.738 0.318 0.623 0.682)
    **12/13 epoch 线上 0.5189131837830901 (0.620 0.832 0.742 0.320 0.624 0.686) (val_0527 0.666 0.894 0.800 0.339 0.670 0.859)**

_20240521_223415
    基于 _20240520_153439 MultiInputMosaic
    10/13 epoch 线上 0.4801529305281125 (0.600 0.810 0.719 0.219 0.606 0.679)

_20240523_182025
    基于 _20240520_153439 MultiInputMixPadding
    10/13 epoch 线上 0.5153156250599193 (0.610 0.824 0.731 0.316 0.613 0.678)
    11/13 epoch 线上 0.5163167392169172 (0.612 0.827 0.734 0.319 0.616 0.678)
    13/13 epoch 线上 0.5161609162894628 (0.613 0.828 0.735 0.321 0.617 0.683)

_20240526_000414[git 未提交]
    基于 _20240520_153439 调小scale -> (0.8, 1.2)
    9/13 epoch 线上 0.510769171502247 (0.617 0.828 0.738 0.292 0.623 0.662) 哪个类别低了？
    12/13 epoch 线上 0.5067752237317477 (0.624 0.835 0.744 0.304 0.629 0.667) 过拟合。(val_0527 0.660 0.890 0.793 0.340 0.662 0.857)

_20240527_101247
    基于 _20240520_153439. train_0527 *验证集同样是_20240518_234754，不太具有参考意义。*
    9/13 epoch 线上 -- (val_updated 0.597 0.800 0.715 0.303 0.599 0.680) (val_0527 0.672 0.901 0.808 0.345 0.673 0.862)
    **10/13 epoch 线上 0.5196930767516853 (val_updated 0.599 0.803 0.719 0.302 0.601 0.680) (val_0527 0.672 0.901 0.810 0.344 0.673 0.862)**
    12/13 epoch 线上 0.5194134889532678 (val_updated 0.600 0.803 0.719 0.303 0.602 0.681) (val_0527 0.672 0.901 0.809 0.336 0.674 0.863)

_20240528_194827
    基于 _20240527_101247 和 _20240523_182025 验证集更新; 两阶段pipeline: MultiInputMixPadding(*7) + 常规pipeline
    9/13 epoch 线上 0.5207424874570967 (val_0527 0.666 0.898 0.806 0.338 0.669 0.860)
    **10/13 epoch 线上 0.5228739374174444 (val_0527 0.667 0.898 0.807 0.342 0.670 0.860)** van: 0.07756455240956982 * 5 = 0.3878227620478491
    12/13 epoch 线上 0.5225812324246998 (val_0527 0.668 0.899 0.809 0.348 0.671 0.862)

_20240530_092722 (mean_fuse)
    MultiInputMosaic*6 + 常规pipeline
    11/13 epoch 线上 0.5202136765716092 (val_0527 0.667 0.896 0.803 0.334 0.669 0.863)
    13/13 epoch 线上 0.5200951346653668 (val_0527 0.670 0.898 0.806 0.337 0.672 0.865)

_20240531_141605_fold_x
    基于 _20240528_194827
    fold0: 12epoch 线上 0.5190668677895692 (val_0527 0.666 0.898 0.803 0.330 0.668 0.863)
           10 epoch 线上 0.5170592197953877(=0.1788830576050569 + 0.3381761621903308) (val_0527 0.664 0.896 0.800 0.307 0.666 0.862)

20240601
    ensemble:
    _20240528_194827(epoch10) + _20240530_092722(epoch11) + _20240531_141605_fold_0(epoch12)
    skip/nms = 0.05/0.7
    线下大概 0.6838240481705212 or (0.674291 0.902451 0.811669 0.350855 0.675636 0.865489)
    线上0.5316742459794518

    skip/nms = 0.11/0.65
    线下大概 0.6930203193886105 or (0.673144 0.901855 0.810492 0.350491 0.674765 0.863098) 
    线上 0.5286825228260283. 可能还是COCO的准

20240602
    ensemble:
    skip/nms = 0.07/0.8
    work_dirs/mean_fuse/_20240530_092722/epoch_11_submit.pkl \
    work_dirs/codetr_all_in_one/_20240531_141605_fold_0/epoch_12_submit.pkl \
    work_dirs/codetr_all_in_one/_20240528_194827/epoch_10_submit.pkl \
    work_dirs/codetr_all_in_one/_20240520_153439/epoch_12_submit.pkl \
    work_dirs/codetr_all_in_one/_20240531_141605_fold_1/epoch_11_submit.pkl \
    线下大概 0.687434782 or (0.676031864	0.90358325	0.815278186	0.353419017	0.677676455	0.865931577)
    线上 0.534338450353712

    线下 0.690895491 or (0.675792044	0.904133767	0.81357435	0.353456609	0.677401048	0.864757538)
    线上 0.5330973930313206


20240603
    ensemble:
    0.07/0.75
    ```
    python scripts/ensemble.py -p \
        work_dirs/codetr_all_in_one/_20240520_153439/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240528_194827/epoch_10_submit.pkl \
        work_dirs/mean_fuse/_20240530_092722/epoch_11_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_0/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_1/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_2/epoch_12_submit.pkl \
        -o ensemble_exp/submit_`date +"%m%d_%H%M%S"`
    ```
    线下大概 0.688424174 (0.677093871	0.905373239	0.816539492	0.352021417	0.678945491	0.864768281)
    线上 0.5344882749788099

    _20240531_141605_fold_2/epoch_12_submit 线上 0.52159267842541
    0.11/0.7 线上 0.5323533461235721


20240604
    ensemble:
    0.07/0.75
    ```
    python scripts/ensemble.py -p \
        work_dirs/codetr_all_in_one/_20240520_153439/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240528_194827/epoch_10_submit.pkl \
        work_dirs/mean_fuse/_20240530_092722/epoch_11_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_0/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_1/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_2/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_3/epoch_12_submit.pkl \
        -o ensemble_exp/submit_`date +"%m%d_%H%M%S"`
    ```
    线下大概 0.687961541 (0.677108555	0.906284883	0.816691961	0.352587451	0.678711164	0.865270554)
    线上 0.5339383937498916


20240605
    ensemble:
    0.07/0.75
    ```
    python scripts/ensemble.py -p \
        work_dirs/codetr_all_in_one/_20240520_153439/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240528_194827/epoch_10_submit.pkl \
        work_dirs/mean_fuse/_20240530_092722/epoch_11_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_0/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_1/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_2/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_4/epoch_12_submit.pkl \
        -o ensemble_exp/submit_`date +"%m%d_%H%M%S"`
    ```
    线上 0.5334507961984957
    _20240531_141605_fold_4/epoch_12_submit.pkl 线上 0.5126251971559532 <最低分>


20240607
    ```
    python scripts/ensemble.py -p \
        work_dirs/codetr_all_in_one/_20240520_153439/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240528_194827/epoch_10_submit.pkl \
        work_dirs/mean_fuse/_20240530_092722/epoch_11_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_0/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_1/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_2/epoch_12_submit.pkl \
        work_dirs/mean_fuse/_20240607_092457/epoch_11_submit.pkl \
        -o ensemble_exp/submit_`date +"%m%d_%H%M%S"`
    ```
    线上 0.5356947630570912

0608
    python scripts/ensemble.py -p \
        work_dirs/codetr_all_in_one/_20240520_153439/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240528_194827/epoch_10_submit.pkl \
        work_dirs/mean_fuse/_20240530_092722/epoch_11_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_0/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_1/epoch_12_submit.pkl \
        work_dirs/codetr_all_in_one/_20240531_141605_fold_2/epoch_12_submit.pkl \
        work_dirs/mean_fuse/_20240607_092457/epoch_11_submit.pkl \
        work_dirs/codetr_all_in_one/_20240607_222820/epoch_12_submit.pkl \
        -o ensemble_exp/submit_`date +"%m%d_%H%M%S"`

    线上 0.5372200160553522

## TODO
conf-thres=0    mosaic + 多模态: 1epoch     线上 ?
conf-thres=0    mosaic + 多模态: ~5epoch    线上 

conf-thres=0    多模态: x epoch 增广rgb     线上 ?


## environment

```
conda create --name gaiic python=3.10 mamba -y
conda activate gaiic

mamba install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia/label/cuda-11.8.0 -y
mamba install openmim einops wandb seaborn -y
pip install fairscale scikit-learn
pip install ensemble-boxes
pip install transformers     # glip

mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet
```