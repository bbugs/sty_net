import numpy as np


def precision_recall_f1(y_pred, y_true, raw_scores=False):
    """(np.array, np.array) -> float, float, float

    Inputs:
    - y_pred: array of size (n_imgs, n_words) of +1 or -1
      y_pred[i,j] indicates our prediction of whether word j goes with
      image i
    - y_true: array of same size as y_pred
      indicates which words are actually present for each image (+1 or -1)

    Returns:
    if raw_scores is False, return:
    - precision: P = tp/(tp + fp)
    - recall: R = tp/(tp + fn)
    - f1 = 2 * P * R / (P + R)

    if raw_scores is True, return:
    - true positives
    - false positives
    - false negatives

                Retrieved
                 Yes  No
    Relevant Yes| tp | fn |
      (true) No | fp | fn |

    """
    # True Positives
    # you predict +1 and it's actually +1
    true_pos = np.zeros(y_pred.shape)
    true_pos[np.logical_and((y_true == 1), (y_pred == 1))] = 1
    true_pos = np.sum(true_pos)

    # False Positives
    # you predict +1 but it's actually -1
    false_positives = np.zeros(y_pred.shape)
    false_positives[np.logical_and((y_true == -1), (y_pred == 1))] = 1
    false_positives = np.sum(false_positives)

    # False Negatives
    # you predict -1 but it's actually +1
    false_negatives = np.zeros(y_pred.shape)
    false_negatives[np.logical_and((y_true == 1), (y_pred == -1))] = 1
    false_negatives = np.sum(false_negatives)

    if raw_scores: # return tp, fp, fn
        return true_pos, false_positives, false_negatives

    precision = float(true_pos) / (true_pos + false_positives)
    recall = float(true_pos) / (true_pos + false_negatives)

    if (precision + recall) == 0:
        f1 = 0.
    else:
        f1 = 2. * precision * recall / (precision + recall)

    return precision, recall, f1


def reciprocal_rank(ytrue, ypred, verbose=False):
    """(int, lst) -> float
    ytrue = 3
    ypred = [5, 2, 4, 3, 0, 1]
    output = 1/4
    """
    assert len(ypred) > 0

    if verbose:
        if (ypred.index(ytrue) + 1) < 5:
            print "ytrue", ytrue
            print "rank", ypred.index(ytrue) + 1

    try:
        rank = ypred.index(ytrue) + 1
    except ValueError:
        return 0.

    return 1. / rank

def mean_reciprocal_rank(ytrue_list, ypred_list, verbose=False):
    """(lst, lst) -> float
    ytrue_list = [2, 0, 1]
    ypred_list = [[0,2,1], [2,1,0], [1,2,0]]
    output = (1/2 + 1/3 + 1)/3 = 0.6111
    """
    assert len(ypred_list) == len(ytrue_list)
    assert len(ytrue_list) > 0
    MRR = 0.

    indx = 0
    for ytrue in ytrue_list:
        MRR += reciprocal_rank(ytrue, ypred_list[indx], verbose)
        indx += 1
    MRR /= len(ytrue_list)

    return MRR


def precision_at_k(ytrue, ypred, k, verbose=False):
    """(lst, lst, int) -> float

    """

    if len(ytrue) == 0:
        return None

    ypred = set(ypred[0:k])
    ytrue = set(ytrue)

    if verbose:
        print "ypred", ypred
        print "ytrue", ytrue
        print "\n"

    return float(len(set.intersection(ypred, ytrue))) / len(ypred)


def recall_at_k(ytrue, ypred, k):
    """(lst, lst, int) -> float

    """
    if len(ytrue) == 0:
        return None

    ypred = set(ypred[0:k])
    ytrue = set(ytrue)
    return float(len(set.intersection(ypred, ytrue))) / len(ytrue)


def f1_score(ytrue, ypred, k):
    """(lst, lst, int) -> float

    """
    if len(ytrue) == 0:
        return None
    P = precision_at_k(ytrue, ypred, k)
    R = recall_at_k(ytrue, ypred, k)

    if P == 0 or R == 0:
        return 0.0

    if (P + R) == 0:
        return None

    return float(2 * (P * R) / (P + R))


def avg_metric_at_k(metric, ytrue_list, ypred_list, k, verbose=False):
    """ (list of lists, list of lists) -> float

    """
    assert len(ytrue_list) == len(ypred_list)
    idx = 0
    counter = 0
    avg_metric = 0.
    for ytrue in ytrue_list:
        ypred = ypred_list[idx]

        metric_at_k = metric(ytrue, ypred, k)
        idx += 1
        # check that prec_at_k is not None
        if metric_at_k is None:
            continue
        counter += 1
        avg_metric += metric_at_k

    if counter == 0:
        return None

    if verbose:
        print "idx, counter, metric:", idx, counter, metric.__name__

    return avg_metric / counter



if __name__ == '__main__':
    true = [4, 9, 10, 13, 14, 24, 37, 39, 76, 90]
    predicted1 = [4, 10, 13, 37, 100, 198, 200]

    k = 5
    p = precision_at_k(true, predicted1, k)
    r = recall_at_k(true, predicted1, k)
    f1 = f1_score(true, predicted1, k)
    print "precision", p
    print "recall", r
    print "f1", f1


    true_list = [[2, 3], [], [], []]
    predicted1_list = [[4, 5], [], [], []]

    avg_prec = avg_metric_at_k(precision_at_k, true_list, predicted1_list, k)
    avg_recall = avg_metric_at_k(recall_at_k, true_list, predicted1_list, k)
    avg_f1 = avg_metric_at_k(f1_score, true_list, predicted1_list, k)

    print "avg_prec", avg_prec
    print "avg_recall", avg_recall
    print "avg_f1", avg_f1




