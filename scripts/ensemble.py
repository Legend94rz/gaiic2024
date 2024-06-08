import argparse
import json
import pickle as pkl
from projects.gaiic2014.core.metrics import map_score
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pandas as pd
import numpy as np
from torchvision.ops import box_convert, box_iou
import torch
import copy
from pathlib import Path
from ensemble_boxes import weighted_boxes_fusion, weighted_boxes_fusion_experimental


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--pred_pkls", nargs='+', type=str)
    # parser.add_argument('-s', "--skip_scores", type=float, default=0.15)
    parser.add_argument('-a', "--annotation", type=str)
    parser.add_argument("-o", "--output", type=str)
    return parser.parse_args()


def read_pred(f):
    # return: {file_id: {'bboxes', 'scores', 'labels'}}
    if f.endswith('pkl'):
        with open(f, 'rb') as fin:
            pred = pkl.load(fin)
        res = {}
        for x in pred:
            inst = x['pred_instances']
            res[x['img_id']] = {
                'labels': inst['labels'].numpy(),
                'bboxes': inst['bboxes'].numpy(),   # xyxy
                'scores': inst['scores'].numpy()
            }
    elif f.endswith('json'):
        with open(f, 'r') as fin:
            pred = json.load(fin)
        df = pd.DataFrame(pred)
        df = df.groupby('image_id')[['bbox', 'score', 'category_id']].agg(list).sort_values('score', ascending=False)
        res = df.to_dict('index')
        for k, v in res.items():
            res[k] = {
                'labels': np.array(v['category_id']) - 1,
                'bboxes': box_convert(torch.tensor(v['bbox']), 'xywh', 'xyxy').numpy(),
                'scores': np.array(v['score']),
            }
    return res


def to_std_format(dct):
    "res: {img_id: {'bbox': ndarray, 'scores': ndarray, 'labels': ndarray}}"
    res = []
    for img_id, preds in dct.items():
        n = len(preds['bboxes'])
        assert n == len(preds['scores']) == len(preds['labels'])
        box = box_convert(torch.from_numpy(preds['bboxes']), 'xyxy', 'xywh').numpy().tolist()
        for i in range(n):
            res.append({
                "bbox": box[i],
                "score": preds['scores'][i],
                "image_id": img_id,
                "category_id": int(preds['labels'][i] + 1)
            })
    return res


def ensemble(preds, inter_iou=0.8, intra_iou=0.7):
    n_models = len(preds)
    blobs = sum([split_blobs(x, intra_iou) for x in preds])
    

def ensemble_v2(preds, skip_thres, nms_thres):
    wh = np.array([[640, 512, 640, 512.]])
    boxes_list = [x['bboxes'] / wh for x in preds]
    scores_list = [x['scores'] for x in preds]
    labels_list = [x['labels'] for x in preds]
    boxes, scores, labels = weighted_boxes_fusion(
        boxes_list, scores_list, labels_list,
        iou_thr=nms_thres, skip_box_thr=skip_thres,
    )
    return {
        'bboxes': boxes * wh,
        'labels': labels.astype(int),
        'scores': scores
    }


def coco_map(anno, res):
    cocoDt = anno.loadRes(res)
    cocoEval = COCOeval(anno, cocoDt, "bbox")
    cocoEval.params.maxDets = [300, 300, 300]
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats[:6]


"""
# val:
python scripts/ensemble.py -p \
    work_dirs/mean_fuse/_20240530_092722/epoch_11.pkl \
    work_dirs/codetr_all_in_one/_20240531_141605_fold_0/epoch_12.pkl \
    work_dirs/codetr_all_in_one/_20240528_194827/epoch_10.pkl \
    -a data/track1-A/annotations/val_0527.json
    
# test A:
python scripts/ensemble.py -p \
    work_dirs/mean_fuse/_20240530_092722/epoch_11_submit.pkl \
    work_dirs/codetr_all_in_one/_20240531_141605_fold_0/epoch_12_submit.pkl \
    work_dirs/codetr_all_in_one/_20240528_194827/epoch_10_submit.pkl \
    -o ensemble.pkl
"""
if __name__ == "__main__":
    args = parse_args()
    all_preds = [read_pred(f) for f in args.pred_pkls]
    with open(args.pred_pkls[0], 'rb') as fin:
        bkp = pkl.load(fin)
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    anno = None
    if args.annotation:
        anno = COCO(args.annotation)
        for i in range(len(all_preds)):
            pred = to_std_format(all_preds[i])
            p, r, ap, f1 = map_score(
                anno,
                pred,
                plot=False, unique_classes=np.array(range(5)), save_dir='confusion_mat', names=('car', 'truck', 'bus', 'van', 'freight_car')
            )
            stats = coco_map(anno, pred)
            print(f'======{args.pred_pkls[i]}=======')
            print(ap.mean(1), ap.mean(), stats)
            print('==============================')
            
    all_img_id = set.union(*[set(x.keys()) for x in all_preds])
    all_preds = {
        k: [x.get(k) for x in all_preds]
        for k in all_img_id
    }
    
    opt_score = 0
    opt_cfg = None
    records = []
    # for skip_thres in np.arange(0.02, 0.15, 0.01):
    #     for nms_thres in np.arange(0.6, 0.96, 0.05):
    for skip_thres in [0.07]:
        for nms_thres in [0.75]:
            final_res = to_std_format({k: ensemble_v2(preds, skip_thres, nms_thres) for k, preds in all_preds.items()})
            if anno is not None:
                p, r, ap, f1 = map_score(
                    anno,
                    final_res,
                    plot=False, unique_classes=np.array(range(5)), save_dir='confusion_mat', names=('car', 'truck', 'bus', 'van', 'freight_car')
                )
                stats = coco_map(anno, final_res)
                print('==============================')
                print(f"the score under setting: skip_thres={skip_thres}, nms_thres={nms_thres} is:")
                print(ap.mean(1), ap.mean(), stats)
                records.append([skip_thres, nms_thres, ap.mean(), *stats])
                if opt_score < ap.mean():
                    opt_score = ap.mean()
                    opt_cfg = [skip_thres, nms_thres]
                print('==============================')
    if opt_cfg is not None:
        print(f"{opt_cfg}: {opt_score}")
        if output_dir:
            pd.DataFrame(records, columns=['skip_thres', 'nms_thres', 'score'] + [f'coco_{j}' for j in range(6)]).to_csv(output_dir / 'grid_search.csv', index=False)
        skip_thres, nms_thres = opt_cfg
    res_per_img = {k: ensemble_v2(preds, skip_thres, nms_thres) for k, preds in all_preds.items()}
    if output_dir:
        for i in range(len(bkp)):
            v = res_per_img[bkp[i]['img_id']]
            idx = np.argsort(-v['scores'])
            bkp[i]['pred_instances'] = {
                'bboxes': torch.tensor(v['bboxes'][idx]),
                'labels': torch.tensor(v['labels'][idx]),
                'scores': torch.tensor(v['scores'][idx]),
            }
        with open(output_dir / "ensemble.pkl", 'wb') as fout:
            pkl.dump(bkp, fout)
            print(f"output pkl saves to: {args.output}, using cfg: {opt_cfg}")
            
    print(f'max #box: {max( len(x["labels"]) for x in res_per_img.values() )}')
    if anno is not None:
        # double-check using COCO API:
        final_res = to_std_format(res_per_img)
        coco_map(anno, final_res)
