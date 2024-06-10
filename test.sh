#!/bin/bash
input_dir=$1
data_root=$2
output_path=$3
ckpt_folder=/home/mw/input/gaiic2024_9ckpt3092/model_data
opt1="test_dataloader={'batch_size': 2, 'num_workers': 8, 'dataset': {'ann_file': 'annotations/test.json', 'data_root': '${data_root}', 'data_prefix': {'img_path': '$4/rgb', 'tir_path': '$4/tir'}}}"
opt2="test_evaluator={'ann_file': '${data_root}/annotations/test.json'}"

mkdir /home/mw/project/cache -p
export MPLCONFIGDIR=/home/mw/project/cache
export PYTORCH_KERNEL_CACHE_PATH=/home/mw/project/cache

python scripts/make_test_json_input.py "${input_dir}" --save_path "${data_root}"/annotations/test.json

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/codetr_all_in_one.py \
${ckpt_folder}/codetr_0527fold0.pth \
--out tmp/codetr_0527fold0.pkl --cfg-options "${opt1}" "${opt2}"

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/codetr_all_in_one.py \
${ckpt_folder}/codetr_0527fold1.pth \
--out tmp/codetr_0527fold1.pkl --cfg-options "${opt1}" "${opt2}"

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/codetr_all_in_one.py \
${ckpt_folder}/codetr_0527fold2.pth \
--out tmp/codetr_0527fold2.pkl  --cfg-options "${opt1}" "${opt2}"

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/codetr_all_in_one.py \
${ckpt_folder}/codetr_full_0518data.pth \
--out tmp/codetr_full_0518data.pkl --cfg-options "${opt1}" "${opt2}"

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/codetr_all_in_one_strong_aug_with_pretrain.py \
${ckpt_folder}/codetr_full_0527data_strong_aug_with_pretrain.pth \
--out tmp/codetr_full_0527data_strong_aug_with_pretrain.pkl --cfg-options "${opt1}" "${opt2}"

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/codetr_all_in_one_strong_aug.py \
${ckpt_folder}/codetr_full_0527data_strong_aug.pth \
--out tmp/codetr_full_0527data_strong_aug.pkl --cfg-options "${opt1}" "${opt2}"

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/codetr_all_in_one.py \
${ckpt_folder}/codetr_full_0527data.pth \
--out tmp/codetr_full_0527data.pkl --cfg-options "${opt1}" "${opt2}"

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/mean_fuse.py \
${ckpt_folder}/mean_fuse_full.pth \
--out tmp/mean_fuse_full.pkl --cfg-options "${opt1}" "${opt2}"

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/mean_fuse_with_pretrained.py \
${ckpt_folder}/mean_fuse_with_pretrained.pth \
--out tmp/mean_fuse_with_pretrained.pkl --cfg-options "${opt1}" "${opt2}"


PYTHONPATH=. python scripts/ensemble.py -p \
    tmp/codetr_0527fold0.pkl \
    tmp/codetr_0527fold1.pkl \
    tmp/codetr_0527fold2.pkl \
    tmp/codetr_full_0518data.pkl \
    tmp/codetr_full_0527data_strong_aug_with_pretrain.pkl \
    tmp/codetr_full_0527data_strong_aug.pkl \
    tmp/codetr_full_0527data.pkl \
    tmp/mean_fuse_full.pkl \
    tmp/mean_fuse_with_pretrained.pkl \
    -o tmp/

PYTHONPATH=. python scripts/submit.py -i ${data_root}/annotations/test.json -r tmp/ensemble.pkl -o ${output_path}
