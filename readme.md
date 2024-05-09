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
[] *验证集加入训练*
[] 输出后处理：如先确定recall接近1时的conf-thres, 在这附近，找一个conf-下降最快的点，低于该点的框置0（或者不输出）. 另外可能有面积极小甚至为0的框，需要去掉。
[] weighted box fusion。官方实现很可能有bug，需要自己写

**WARN**
由于B阶段只有一天时间，两次提交机会。每次提交之后一定注意要保存最优的模型权重，一旦丢失，当天是来不及重新训练的。
在比赛过程中注意保存最优的两个权重。同时A阶段结束之前3天(6月7号)再次检查是否缺失。

**WARN**
2024年5月17日8:00之前发邮件到大赛邮箱data@tsinghua.edu.cn
使用的模型权重（下载链接、md5）及数据集


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
    多模态, const lr + cos annealing, 10/13epoch, randshift+AdaptiveHistEQU+swa+concat deform attn: 线上 0.4901899498048514 (0.644 0.866 0.769 0.322 0.647 0.831) tta 0.4683023883758988 可能参数没调好，需要检查下结果，并且tta可能超时(1h)
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
pip install fairscale -y
pip install ensemble-boxes -y
pip install transformers -y     # glip

mim install mmengine -y
mim install "mmcv>=2.0.0" -y
mim install mmdet -y
```