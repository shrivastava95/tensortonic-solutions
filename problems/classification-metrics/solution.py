from itertools import product
import numpy as np

def _get_accuracy(tp,fp,fn):
    return (2 * tp) / (2 * tp + fp + fn)

def _get_precision(tp,fp,fn):
    return tp / (tp + fp)

def _get_recall(tp,fp,fn):
    return tp / (tp + fn)

def _get_f1(tp,fp,fn):
    p = _get_precision(tp,fp,fn)
    r = _get_recall(tp,fp,fn)
    return (2 * p * r) / (p + r)

def classification_metrics(y_true, y_pred, average="micro", pos_label=1):
    """
    Compute accuracy, precision, recall, F1 for single-label classification.
    Averages: 'micro' | 'macro' | 'weighted' | 'binary' (uses pos_label).
    Return dict with float values.
    """
    y, p = list(map(np.array, [y_true, y_pred]))
    C = max(list(list(np.unique(y)) + list(np.unique(p)))) + 1
    N = len(y)
    real_pred = np.zeros([C, C]).astype(np.float64)
    for yi, pi in zip(y, p):
        real_pred[yi, pi] += 1
    
    match average:
        case "micro":
            # i guess that we do not compute TN because that would just be inda silly.
            tp, fp, fn = 0, 0, 0
            for i, j in product(range(C), range(C)):
                if i == j:
                    tp += real_pred[i, j]
                elif i != j:
                    fp += real_pred[i, j]
                    fn += real_pred[i, j]
            accuracy = _get_accuracy(tp,fp,fn)           
            precision = _get_precision(tp,fp,fn)
            recall = _get_recall(tp,fp,fn)
            f1 = _get_f1(tp,fp,fn)
            metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}
        case "macro":
            cwmetrics = {"accuracy": [0]*C, "precision": [0]*C, "recall": [0]*C, "f1": [0]*C}
            tp, fp, fn = [[0]*C, [0]*C, [0]*C]
            
            for i, j, ci in product(range(C), range(C), range(C)):
                if ci == j or ci == i:
                    if ci == j and ci == i: # tp
                        tp[ci] += real_pred[i, j]
                    elif ci == j and ci != i: # fp
                        fp[ci] += real_pred[i, j]
                    elif ci != j and ci == i: # fn
                        fn[ci] += real_pred[i, j]
            # classwise metrics
            for ci in range(C):
                tpi, fpi, fni = [item[ci] for item in [tp, fp, fn]]
                cwmetrics["accuracy"][ci] = _get_accuracy(tpi,fpi,fni)
                cwmetrics["precision"][ci] = _get_precision(tpi,fpi,fni)
                cwmetrics["recall"][ci] = _get_recall(tpi,fpi,fni)
                cwmetrics["f1"][ci] = _get_f1(tpi,fpi,fni)

                for k, v in cwmetrics.items():
                    cwmetrics[k] = np.mean(v).item()
            metrics = cwmetrics
        case _:
            return None

    return metrics
