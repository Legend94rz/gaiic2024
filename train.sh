#!/bin/bash
export OMP_NUM_THREADS=1
export PYTHONPATH=.

data_root=data
nnode=1

# 数据路径整理(为了避免修改代码)
pushd data/track1-A
ln -s ../contest_data/train train
ln -s ../contest_data/val val
ln -s ../contest_data/test test
popd

echo "========== section 1. data & pretrained ckpts =============="
# 下载权重 swin/co-dino
mkdir -p ckpt
pushd ckpt
wget https://download.openmmlab.com/mmdetection/v3.0/codetr/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth
wget https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth
popd

# patch ckpt
python scripts/patch_ckpt.py --ckpt ckpt/co_dino_5scale_swin_large_16e_o365tococo-614254c9.pth

# split nfold
python scripts/split_nfold.py ${data}/track1-A/annotations/train_0527.json --prefix train_0527


echo "========== section 3. train models using contest data =============="
# train model [1]: codetr [train_0518 full data] [concat]
cfg=projects/gaiic2014/configs/codetr_full_0518data.py
opt="train_dataloader={'dataset': {'ann_file': 'annotations/train_0518.json'}}"
torchrun --nproc-per-node=${nnode} tools/train.py ${cfg} --launcher pytorch --work-dir data/tmp_data/work_dirs/codetr_all_in_one_0518/ --cfg-options "${opt}"

# train model [2]: codetr [train_0527 full data] [concat]
cfg=projects/gaiic2014/configs/codetr_all_in_one.py
torchrun --nproc-per-node=${nnode} tools/train.py ${cfg} --launcher pytorch --work-dir data/tmp_data/work_dirs/codetr_all_in_one_0527/

# train model [3/4/5]: codetr [train_0527 0/1/2 fold] [concat]
for i in {0..2}
do
    cfg=projects/gaiic2014/configs/codetr_all_in_one.py
    opt="train_dataloader={'dataset': {'ann_file': 'annotations/train_0527_${i}_train.json'}}"
    opt2="train_cfg={'max_epochs': 12}"
    torchrun --nproc-per-node=${nnode} tools/train.py ${cfg} --launcher pytorch --work-dir data/tmp_data/work_dirs/codetr_all_in_one_fold_${i}/ --cfg-options "${opt}" "${opt2}"
done

# train model [6]: codetr [train_0527 full data] [strong aug] [concat]
cfg=projects/gaiic2014/configs/codetr_all_in_one_strong_aug.py
torchrun --nproc-per-node=${nnode} tools/train.py ${cfg} --launcher pytorch --work-dir data/tmp_data/work_dirs/codetr_all_in_one_strong_aug/

# train model [7]: codetr [train_0527 full data] [mean]
cfg=projects/gaiic2014/configs/mean_fuse.py
torchrun --nproc-per-node=${nnode} tools/train.py ${cfg} --launcher pytorch --work-dir data/tmp_data/work_dirs/mean_fuse/

echo "========== section 4. train models with EXTERNAL DATA =============="
echo "由于外部数据不能完全自动化，不好写到脚本里。这部分如果需要严格复现，可以手动取消注释。外部数据的模型A榜约有5个千分点的提高。对最终名次无影响。"
# # aistudio data
# echo "手动下载aistudio数据集, 解压到 data 目录, 最终data目录结构请参考 readme"
# echo "下载链接: https://aistudio.baidu.com/datasetdetail/206266/0"

# # vedai data
# pushd ${data_root}
# mkdir vedai
# cd vedai
# wget https://downloads.greyc.fr/vedai/Annotations512.tar
# wget https://downloads.greyc.fr/vedai/Vehicules512.tar.001
# wget https://downloads.greyc.fr/vedai/Vehicules512.tar.002
# cat Vehicules512.tar* | tar xvf
# tar -xvf Annotations512.tar
# popd

# # format external data
# python scripts/external_data.py     # [WARN] 注意这个脚本里的路径都写死了，所以数据的目录树要与readme里保持一致
# python scripts/split_nfold.py ${data}/aistudio/annotations.json       # 用aistudio的其中一折作为验证集

# # pre-train: codetr [EXTERNAL DATA] [mean]
# cfg=projects/gaiic2014/configs/pretrain.py
# torchrun --nproc-per-node=${nnode} tools/train.py ${cfg} --launcher pytorch --work-dir data/tmp_data/work_dirs/pretrain/

# # patch the trained model ckpt
# cp data/tmp_data/work_dirs/pretrain/epoch_10.pth ckpt/codino_pretrained_240607_epoch10.pth
# python scripts/patch_ckpt.py --ckpt ckpt/codino_pretrained_240607_epoch10.pth

# # train model [8]: codetr [pretrained] [train_0527 full data] [mean]
# cfg=projects/gaiic2014/configs/mean_fuse_with_pretrained.py
# torchrun --nproc-per-node=${nnode} tools/train.py ${cfg} --launcher pytorch --work-dir data/tmp_data/work_dirs/mean_fuse_with_pretrained/

# # train model [9]: codetr [pretrained] [train_0527 full data] [strong aug] [concat]
# cfg="projects/gaiic2014/configs/codetr_all_in_one_strong_aug_with_pretrain.py"
# torchrun --nproc-per-node=${nnode} tools/train.py ${cfg} --launcher pytorch --work-dir data/tmp_data/work_dirs/codetr_all_in_one_strong_aug_with_pretrain/


echo "========== section 5. trim ckpt files =============="
# 移除不需要的字段，减小文件体积，以符合线上提交要求。
python scripts/trim_ckpt.py     data/tmp_data/work_dirs/codetr_all_in_one_0518/epoch_12.pth                       data/tmp_data/codetr_full_0518data.pth
python scripts/trim_ckpt.py     data/tmp_data/work_dirs/codetr_all_in_one_0527/epoch_10.pth                       data/tmp_data/codetr_full_0527data.pth
python scripts/trim_ckpt.py     data/tmp_data/work_dirs/codetr_all_in_one_fold_0/epoch_12.pth                     data/tmp_data/codetr_0527fold0.pth
python scripts/trim_ckpt.py     data/tmp_data/work_dirs/codetr_all_in_one_fold_1/epoch_12.pth                     data/tmp_data/codetr_0527fold1.pth
python scripts/trim_ckpt.py     data/tmp_data/work_dirs/codetr_all_in_one_fold_2/epoch_12.pth                     data/tmp_data/codetr_0527fold2.pth
python scripts/trim_ckpt.py     data/tmp_data/work_dirs/codetr_all_in_one_strong_aug/epoch_12.pth                 data/tmp_data/codetr_full_0527data_strong_aug.pth
python scripts/trim_ckpt.py     data/tmp_data/work_dirs/mean_fuse/epoch_11.pth                                    data/tmp_data/mean_fuse_full.pth

# python scripts/trim_ckpt.py     data/tmp_data/work_dirs/mean_fuse_with_pretrained/epoch_11.pth                    data/tmp_data/mean_fuse_with_pretrained.pth
# python scripts/trim_ckpt.py     data/tmp_data/work_dirs/codetr_all_in_one_strong_aug_with_pretrain/epoch_12.pth   data/tmp_data/codetr_full_0527data_strong_aug_with_pretrain.pth
echo "DONE"
