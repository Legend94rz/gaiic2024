[] 线上线下差距较大的问题(是否与score阈值有关? maxDet?): 不要加阈值过滤。仍然没有完全解决：需要确认下是否是val与test分布差距较大，看下预测结果。
[] 数据可视化；预测结果 badcase
[x] 排查训练速度问题
[x] test json
[x] 检查下各种resize/crop参数的正确性，w/h的顺序
[x] 检查下各种bbox的格式：到底是min/max还是min/wh还是cxy/wh: train/val/test pkl/output json.
    标注与提交都是min/wh
[x] 验证单模态(tir/rgb)的分数 (as baseline)
[] 验证mosaic对于CoDETR的有效性
[] appearance augmentation || fuse augmentaion
[x] wandb。尽快确定改动是否有效
[] giouloss -> ciou loss / siou loss
[] lr warmup & annealing
[] swa

**WARN**
由于B阶段只有一天时间，两次提交机会。每次提交之后一定注意要保存最优的模型权重，一旦丢失，当天是来不及重新训练的。
在比赛过程中注意保存最优的两个权重。同时A阶段结束之前3天(6月7号)再次检查是否缺失。


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
conf-thres=0    多模态: 16 epoch            线上 ?
conf-thres=0    mosaic + 多模态: 1epoch     线上 ?
conf-thres=0    mosaic + 多模态: ~5epoch    线上 

conf-thres=0    多模态: x epoch 增广rgb            线上 ?


