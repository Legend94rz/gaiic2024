import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from torchvision.ops import box_convert, box_iou
from typing import Union


def process_batch(pred_box, pred_cls, gt_box, gt_labels, iouv):
    """
    Return correct predictions matrix. Both sets of boxes are in (x1, y1, x2, y2) format.
    Arguments:
        pred_box, gt_box: (Array[N, 4]), x1, y1, x2, y2
        pred_cls (Array[N, ])
        gt_labels (Array[M, ])
    Returns:
        correct (Array[N, 10]), for 10 IoU levels
    """
    correct = torch.zeros(pred_box.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(gt_box, pred_box)
    i, j = torch.where((iou >= iouv[0]) & (gt_labels.reshape(-1, 1) == pred_cls.reshape(1, -1)))  # IoU above threshold and classes match. x: [#GT, #pred]
    if i.shape[0]:
        matches = torch.cat((torch.stack([i, j], 1), iou[i, j][:, None]), 1).cpu().numpy()  # [label, detection, iou]
        if i.shape[0] > 1:
            matches = matches[matches[:, 2].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        matches = torch.Tensor(matches).to(iouv.device)
        correct[matches[:, 1].long()] = matches[:, 2:3] >= iouv
    return correct


def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves
    # Arguments
        recall:    The recall curve (list)
        precision: The precision curve (list)
    # Returns
        Average precision, precision curve, recall curve
    """

    # Append sentinel values to beginning and end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Compute the precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    # Integrate area under curve
    method = 'interp'  # methods: 'continuous', 'interp'
    if method == 'interp':
        x = np.linspace(0, 1, 101)  # 101-point interp (COCO)
        ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate
    else:  # 'continuous'
        i = np.where(mrec[1:] != mrec[:-1])[0]  # points where x axis (recall) changes
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])  # area under curve

    return ap, mpre, mrec


def plot_mc_curve(px, py, save_dir='mc_curve.png', names=(), xlabel='Confidence', ylabel='Metric'):
    # Metric-confidence curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py):
            ax.plot(px, y, linewidth=1, label=f'{names[i]}')  # plot(confidence, metric)
    else:
        ax.plot(px, py.T, linewidth=1, color='grey')  # plot(confidence, metric)

    y = py.mean(0)
    ax.plot(px, y, linewidth=3, color='blue', label=f'all classes {y.max():.2f} at {px[y.argmax()]:.3f}')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def plot_pr_curve(px, py, ap, save_dir='pr_curve.png', names=()):
    # Precision-recall curve
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:  # display per-class legend if < 21 classes
        for i, y in enumerate(py.T):
            ax.plot(px, y, linewidth=1, label=f'{names[i]} {ap[i, 0]:.3f}')  # plot(recall, precision)
    else:
        ax.plot(px, py, linewidth=1, color='grey')  # plot(recall, precision)

    ax.plot(px, py.mean(1), linewidth=3, color='blue', label='all classes %.3f mAP@0.5' % ap[:, 0].mean())
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(Path(save_dir), dpi=250)
    plt.close()


def ap_per_class(tp, conf, pred_cls, target_cls, unique_classes=None, plot=False, save_dir :Union[str, Path] = '.', names=()):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:  True positives (nparray, nx1 or nx10).
        conf:  Objectness value from 0-1 (nparray).
        pred_cls:  Predicted object classes (nparray).
        target_cls:  True object classes (nparray).
        plot:  Plot precision-recall curve at mAP@0.5
        save_dir:  Plot save directory
    # Returns
        The average precision as computed in py-faster-rcnn.
    # Notes
        len(tp) == len(conf) == len(pred_cls)
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    if unique_classes is None:
        unique_classes = np.unique(target_cls)
    nc = len(unique_classes)  # number of classes, number of detections

    # Create Precision-Recall curve and compute AP for each class
    px, py = np.linspace(0, 1, 1000), []  # for plotting
    ap, p, r = np.zeros((nc, tp.shape[1])), np.zeros((nc, 1000)), np.zeros((nc, 1000))
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = (target_cls == c).sum()  # number of labels
        n_p = i.sum()  # number of predictions

        if n_p == 0 or n_l == 0:
            continue
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum(0)
            tpc = tp[i].cumsum(0)

            # Recall
            recall = tpc / (n_l + 1e-16)  # recall curve
            r[ci] = np.interp(-px, -conf[i], recall[:, 0], left=0)  # negative x, xp because xp decreases. recall[:, 0]: iou-thres=0.5时. r[ci]: 第ci类的recall[:, 0]插值到1000个点.

            # Precision
            precision = tpc / (tpc + fpc)  # precision curve. tpc + fpc == arange(len(tp[i]))
            p[ci] = np.interp(-px, -conf[i], precision[:, 0], left=1)  # p at pr_score

            # AP from recall-precision curve
            for j in range(tp.shape[1]):
                ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
                if plot and j == 0:
                    py.append(np.interp(px, mrec, mpre))  # precision at mAP@0.5

    # Compute F1 (harmonic mean of precision and recall)
    f1 = 2 * p * r / (p + r + 1e-16)
    if plot:
        plot_pr_curve(px, py, ap, Path(save_dir) / 'PR_curve.png', names)
        plot_mc_curve(px, f1, Path(save_dir) / 'F1_curve.png', names, ylabel='F1')
        plot_mc_curve(px, p, Path(save_dir) / 'P_curve.png', names, ylabel='Precision')
        plot_mc_curve(px, r, Path(save_dir) / 'R_curve.png', names, ylabel='Recall')

    i = f1.mean(0).argmax()  # max F1 index
    return p[:, 0], r[:, 0], ap, f1[:, i], unique_classes.astype('int32')


def map_score(cocogt, results, unique_classes, names, plot, save_dir):
    cocodt = cocogt.loadRes(results)
    stats = []
    for img_id in cocogt.getImgIds():
        gtbox = box_convert(torch.tensor([
            x['bbox'] for x in cocogt.loadAnns(cocogt.getAnnIds(img_id))
        ]).reshape(-1, 4), 'xywh', 'xyxy')

        gtlab = torch.tensor([x['category_id'] - 1 for x in cocogt.loadAnns(cocogt.getAnnIds(img_id))])

        predbox = box_convert(torch.tensor([
            x['bbox'] for x in cocodt.loadAnns(cocodt.getAnnIds(img_id))
        ]).reshape(-1, 4), 'xywh', 'xyxy')
        predlab = torch.tensor([x['category_id'] - 1 for x in cocodt.loadAnns(cocodt.getAnnIds(img_id))])
        predscore = torch.tensor([x['score'] for x in cocodt.loadAnns(cocodt.getAnnIds(img_id))])

        correct = process_batch(predbox, predlab, gtbox, gtlab, torch.arange(0.5, 0.99, 0.05))
        stats.append([correct, predscore, predlab, gtlab])
    stats = [np.concatenate(x, 0) for x in zip(*stats)]
    
    p, r, ap, f1, _ = ap_per_class(*stats, plot=True, unique_classes=unique_classes, save_dir=save_dir, names=names)
    return p, r, ap, f1
