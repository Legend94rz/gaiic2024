[x] 排查训练速度问题
[x] test json
[x] 检查下各种resize/crop参数的正确性，w/h的顺序
[x] 检查下各种bbox的格式：到底是min/max还是min/wh还是cxy/wh: train/val/test pkl/output json.
    标注与提交都是min/wh
[] 验证单模态(tir/rgb)的分数 (as baseline)
[] 验证mosaic对于CoDETR的有效性
[x] 线上线下差距较大的问题(是否与score阈值有关? maxDet?): 不要加阈值过滤。maxDet是否有关？
[] appearance augmentation
[] wandb。尽快确定改动是否有效
[] giouloss -> ciou loss / siou loss
[] lr warmup & annealing


速度问题:
注意提交时不要 加--show参数，很慢。
with mosaic:
    backbone with_cp=False, encoder with_cp=6, batchsize=1: OOM (eta: >4 days)
    backbone with_cp=True, encoder with_cp=4,  batchsize=2: OOM (eta: >4 days)

without mosaic:
    [only tir] backbone with_cp=False, encoder with_cp=6, batchsize=3: ? (eta: ~2 days)
    [2 modalities] ... (eta ?)


更小的img size
with_cp
smaller layer / 400 query: 可能会显著影响指标
larger batch_size: 需要改代码。主要是由于mosaic带来的影响，每个batch里有些样本有mosaic，有些没有，导致分辨率不同。简单的改法是mosaic==1.0

#### submit records

conf-thres=0.5, mosaic + 多模态: 1epoch     线上 0.2686912743221043
conf-thres=0.5, 单模态: 1 epoch             线上 0.22698282977757558

conf-thres=0    单模态: 1 epoch             线上 0.38238603322554376
conf-thres=0    单模态: 10 epoch            线上 0.38908813383119695
conf-thres=0    多模态: 1 epoch             线上 ?
conf-thres=0    多模态: x epoch             线上 ?
conf-thres=0    多模态: x epoch 增广rgb            线上 ?


