from sklearn.metrics import auc, precision_recall_curve, roc_curve
import numpy as np

def aupr(preds, labels, pos_label=1):
    """Calculate and return the area under the Precision Recall curve using unthresholded predictions on the data and a binary true label.
    
    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.
    pos_label: label of the positive class (1 by default)
    """
    precision, recall, _ = precision_recall_curve(labels, preds, pos_label=pos_label)
    return auc(recall, precision)

class StatsMeter(object):
    """Computes and stores the average and current value.

    Adapted from: https://github.com/pytorch/examples/blob/master/imagenet/train.py
    """
    def __init__(self):
        self.val = 0.
        self.avg = 0.
        self.sum = 0.
        self.count = 0.
    
    def calculate_stats(self, preds, labels, pos_label=1):
        fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)

        # AUROC score
        auroc = auc(fpr, tpr)

        # AUPRC in score
        precision, recall, _ = precision_recall_curve(labels, preds, pos_label=pos_label)
        aupr_in = auc(recall, precision)

        # FPR@95
        fpr95 = None
        if all(tpr < 0.95):
            # No threshold allows TPR >= 0.95
            fpr95 = 0
        elif all(tpr >= 0.95):
            # All thresholds allow TPR >= 0.95, so find lowest possible FPR
            idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
            fpr95 = min(map(lambda idx: fpr[idx], idxs))
        else:
            # Linear interp between values to get FPR at TPR == 0.95
            fpr95 = np.interp(0.95, tpr, fpr)
        
        return auroc, aupr_in, fpr95