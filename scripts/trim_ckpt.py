import argparse
import torch
from pathlib import Path
"""
python scripts/trim_ckpt.py     work_dirs/codetr_all_in_one/_20240520_153439/epoch_12.pth           model_data/codetr_full_0518data.pth
python scripts/trim_ckpt.py     work_dirs/codetr_all_in_one/_20240528_194827/epoch_10.pth           model_data/codetr_full_0527data.pth
python scripts/trim_ckpt.py     work_dirs/mean_fuse/_20240530_092722/epoch_11.pth                   model_data/mean_fuse_full.pth
python scripts/trim_ckpt.py     work_dirs/mean_fuse/_20240607_092457/epoch_11.pth                   model_data/mean_fuse_with_pretrained.pth
python scripts/trim_ckpt.py     work_dirs/codetr_all_in_one/_20240531_141605_fold_0/epoch_12.pth    model_data/codetr_0527fold0.pth
python scripts/trim_ckpt.py     work_dirs/codetr_all_in_one/_20240531_141605_fold_1/epoch_12.pth    model_data/codetr_0527fold1.pth
python scripts/trim_ckpt.py     work_dirs/codetr_all_in_one/_20240531_141605_fold_2/epoch_12.pth    model_data/codetr_0527fold2.pth
python scripts/trim_ckpt.py     work_dirs/codetr_all_in_one/_20240607_222820/epoch_12.pth           model_data/codetr_full_0527data_strong_aug.pth
python scripts/trim_ckpt.py     work_dirs/codetr_all_in_one_complex_data_with_pretrain/_20240608_201858/epoch_12.pth           model_data/codetr_full_0527data_strong_aug_with_pretrain.pth
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt', type=str)
    parser.add_argument('save_path', type=str)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    x = Path(args.ckpt)
    ckpt = torch.load(x, map_location='cpu')
    ckpt.pop('ema_state_dict', None)
    ckpt.pop('optimizer')
    ckpt.pop('param_schedulers')
    Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, args.save_path)
