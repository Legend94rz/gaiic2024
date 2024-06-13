#!/bin/bash
input_dir=$1
data_root=$2
output_path=$3
ckpt_folder=data/best_model
opt1="test_dataloader={'batch_size': 2, 'num_workers': 8, 'dataset': {'ann_file': 'annotations/test.json', 'data_root': '${data_root}', 'data_prefix': {'img_path': '$4/rgb', 'tir_path': '$4/tir'}}}"
opt2="test_evaluator={'ann_file': '${data_root}/annotations/test.json'}"

python scripts/make_test_json_input.py "${input_dir}" --save_path "${data_root}"/annotations/test.json

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/codetr_all_in_one.py \
${ckpt_folder}/codetr_0527fold0.pth \
--out data/tmp_data/codetr_0527fold0.pkl --cfg-options "${opt1}" "${opt2}"

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/codetr_all_in_one.py \
${ckpt_folder}/codetr_0527fold1.pth \
--out data/tmp_data/codetr_0527fold1.pkl --cfg-options "${opt1}" "${opt2}"

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/codetr_all_in_one.py \
${ckpt_folder}/codetr_0527fold2.pth \
--out data/tmp_data/codetr_0527fold2.pkl  --cfg-options "${opt1}" "${opt2}"

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/codetr_all_in_one.py \
${ckpt_folder}/codetr_full_0518data.pth \
--out data/tmp_data/codetr_full_0518data.pkl --cfg-options "${opt1}" "${opt2}"

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/codetr_all_in_one_strong_aug_with_pretrain.py \
${ckpt_folder}/codetr_full_0527data_strong_aug_with_pretrain.pth \
--out data/tmp_data/codetr_full_0527data_strong_aug_with_pretrain.pkl --cfg-options "${opt1}" "${opt2}"

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/codetr_all_in_one_strong_aug.py \
${ckpt_folder}/codetr_full_0527data_strong_aug.pth \
--out data/tmp_data/codetr_full_0527data_strong_aug.pkl --cfg-options "${opt1}" "${opt2}"

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/codetr_all_in_one.py \
${ckpt_folder}/codetr_full_0527data.pth \
--out data/tmp_data/codetr_full_0527data.pkl --cfg-options "${opt1}" "${opt2}"

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/mean_fuse.py \
${ckpt_folder}/mean_fuse_full.pth \
--out data/tmp_data/mean_fuse_full.pkl --cfg-options "${opt1}" "${opt2}"

WANDB_MODE=offline PYTHONPATH=. python tools/test.py \
projects/gaiic2014/configs/mean_fuse_with_pretrained.py \
${ckpt_folder}/mean_fuse_with_pretrained.pth \
--out data/tmp_data/mean_fuse_with_pretrained.pkl --cfg-options "${opt1}" "${opt2}"


PYTHONPATH=. python scripts/ensemble.py -p \
    data/tmp_data/codetr_0527fold0.pkl \
    data/tmp_data/codetr_0527fold1.pkl \
    data/tmp_data/codetr_0527fold2.pkl \
    data/tmp_data/codetr_full_0518data.pkl \
    data/tmp_data/codetr_full_0527data_strong_aug_with_pretrain.pkl \
    data/tmp_data/codetr_full_0527data_strong_aug.pkl \
    data/tmp_data/codetr_full_0527data.pkl \
    data/tmp_data/mean_fuse_full.pkl \
    data/tmp_data/mean_fuse_with_pretrained.pkl \
    -o data/tmp_data/

PYTHONPATH=. python scripts/submit.py -i ${data_root}/annotations/test.json -r data/tmp_data/ensemble.pkl -o ${output_path}
