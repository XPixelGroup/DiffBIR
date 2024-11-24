from typing import List, Tuple

import numpy as np
from numpy import ndarray


def get_mAP(
    preds: ndarray,
    gt_file: str,
    taglist: List[str]
) -> Tuple[float, ndarray]:
    assert preds.shape[1] == len(taglist)

    # When mapping categories from test datasets to our system, there might be
    # multiple vs one situation due to different semantic definitions of tags.
    # So there can be duplicate tags in `taglist`. This special case is taken
    # into account.
    tag2idxs = {}
    for idx, tag in enumerate(taglist):
        if tag not in tag2idxs:
            tag2idxs[tag] = []
        tag2idxs[tag].append(idx)

    # build targets
    targets = np.zeros_like(preds)
    with open(gt_file, "r") as f:
        lines = [line.strip("\n").split(",") for line in f.readlines()]
    assert len(lines) == targets.shape[0]
    for i, line in enumerate(lines):
        for tag in line[1:]:
            targets[i, tag2idxs[tag]] = 1.0

    # compute average precision for each class
    APs = np.zeros(preds.shape[1])
    for k in range(preds.shape[1]):
        APs[k] = _average_precision(preds[:, k], targets[:, k])

    return APs.mean(), APs


def _average_precision(output: ndarray, target: ndarray) -> float:
    epsilon = 1e-8

    # sort examples
    indices = output.argsort()[::-1]
    # Computes prec@i
    total_count_ = np.cumsum(np.ones((len(output), 1)))

    target_ = target[indices]
    ind = target_ == 1
    pos_count_ = np.cumsum(ind)
    total = pos_count_[-1]
    pos_count_[np.logical_not(ind)] = 0
    pp = pos_count_ / total_count_
    precision_at_i_ = np.sum(pp)
    precision_at_i = precision_at_i_ / (total + epsilon)

    return precision_at_i


def get_PR(
    pred_file: str,
    gt_file: str,
    taglist: List[str]
) -> Tuple[float, float, ndarray, ndarray]:
    # When mapping categories from test datasets to our system, there might be
    # multiple vs one situation due to different semantic definitions of tags.
    # So there can be duplicate tags in `taglist`. This special case is taken
    # into account.
    tag2idxs = {}
    for idx, tag in enumerate(taglist):
        if tag not in tag2idxs:
            tag2idxs[tag] = []
        tag2idxs[tag].append(idx)

    # build preds
    with open(pred_file, "r", encoding="utf-8") as f:
        lines = [line.strip().split(",") for line in f.readlines()]
    preds = np.zeros((len(lines), len(tag2idxs)), dtype=bool)
    for i, line in enumerate(lines):
        for tag in line[1:]:
            preds[i, tag2idxs[tag]] = True

    # build targets
    with open(gt_file, "r", encoding="utf-8") as f:
        lines = [line.strip().split(",") for line in f.readlines()]
    targets = np.zeros((len(lines), len(tag2idxs)), dtype=bool)
    for i, line in enumerate(lines):
        for tag in line[1:]:
            targets[i, tag2idxs[tag]] = True

    assert preds.shape == targets.shape

    # calculate P and R
    TPs = ( preds &  targets).sum(axis=0)  # noqa: E201, E222
    FPs = ( preds & ~targets).sum(axis=0)  # noqa: E201, E222
    FNs = (~preds &  targets).sum(axis=0)  # noqa: E201, E222
    eps = 1.e-9
    Ps = TPs / (TPs + FPs + eps)
    Rs = TPs / (TPs + FNs + eps)

    return Ps.mean(), Rs.mean(), Ps, Rs
