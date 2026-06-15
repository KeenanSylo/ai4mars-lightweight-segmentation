import numpy as np
import torch

CLASS_NAMES = ["soil", "bedrock", "sand", "big_rock"]
NUM_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = 255


def update_confusion_matrix(cm, preds, labels, num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    """In-place update of a (num_classes x num_classes) numpy confusion matrix.

    Rows are true classes, columns are predicted classes.
    Pixels where label == ignore_index are excluded.
    """
    valid = labels != ignore_index
    p = preds[valid].astype(np.int64).ravel()
    l = labels[valid].astype(np.int64).ravel()
    idx = l * num_classes + p
    counts = np.bincount(idx, minlength=num_classes * num_classes)
    cm += counts.reshape(num_classes, num_classes)
    return cm


def update_confusion_matrix_torch(cm, preds, labels,
                                  num_classes=NUM_CLASSES, ignore_index=IGNORE_INDEX):
    """In-place update of a (num_classes x num_classes) torch confusion matrix.

    Stays on the same device as `cm` (so train/val accumulation can stay on the
    GPU and avoid a per-batch CPU transfer). `preds` and `labels` must be int
    tensors on the same device as `cm`. Rows = true class, columns = predicted
    class. Pixels where label == ignore_index are excluded.
    """
    valid = labels != ignore_index
    p = preds[valid].long().view(-1)
    l = labels[valid].long().view(-1)
    indices = l * num_classes + p
    binc = torch.bincount(indices, minlength=num_classes * num_classes)
    cm += binc.view(num_classes, num_classes)
    return cm


def compute_metrics(cm):
    """Derive all summary metrics from a (num_classes x num_classes) confusion matrix.

    Returns a dict with per-class arrays (iou, precision, recall, f1, support)
    and scalar aggregates (pixel_acc, miou, macro_precision, macro_recall,
    macro_f1). Aggregates are computed over classes with support > 0.
    """
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    fn = cm.sum(axis=1) - tp
    fp = cm.sum(axis=0) - tp
    eps = 1e-12

    iou = tp / (tp + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    support = cm.sum(axis=1)

    total = cm.sum()
    pixel_acc = tp.sum() / (total + eps)

    valid_classes = support > 0
    if valid_classes.any():
        miou = float(iou[valid_classes].mean())
        macro_precision = float(precision[valid_classes].mean())
        macro_recall = float(recall[valid_classes].mean())
        macro_f1 = float(f1[valid_classes].mean())
    else:
        miou = macro_precision = macro_recall = macro_f1 = 0.0

    return {
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "pixel_acc": float(pixel_acc),
        "miou": miou,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "valid_classes": valid_classes,
    }


def format_report(cm, title=""):
    m = compute_metrics(cm)
    iou, prec, rec, sup = m["iou"], m["precision"], m["recall"], m["support"]
    lines = []
    if title:
        lines.append(f"=== {title} ===")
    lines.append(f"  pixel accuracy : {m['pixel_acc']:.4f}")
    lines.append(f"  mean IoU       : {m['miou']:.4f}  (over labelled classes only)")
    lines.append("")
    lines.append(f"  {'class':<10} {'IoU':>7} {'Prec':>7} {'Recall':>7} {'support':>15}")
    for i, name in enumerate(CLASS_NAMES):
        s = int(sup[i])
        if s == 0:
            lines.append(f"  {name:<10} {'--':>7} {'--':>7} {'--':>7} {s:>15,}  (no test pixels)")
        else:
            lines.append(f"  {name:<10} {iou[i]:7.4f} {prec[i]:7.4f} {rec[i]:7.4f} {s:>15,}")
    return "\n".join(lines)
