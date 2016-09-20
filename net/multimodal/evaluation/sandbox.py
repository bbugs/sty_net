import numpy as np

np.random.seed(4)
scores = np.random.randn(7, 5)
y_pred = np.ones(scores.shape, dtype=int)
y_pred[scores < 0] = -1

y_true = np.random.randint(-1, 1, scores.shape)
y_true[y_true == 0] = 1

print "y_true \n", y_true, "\n"
# print scores, "\n"
print "y_pred \n", y_pred, "\n"

# True Positives
true_pos = np.zeros(scores.shape)
true_pos[np.logical_and((y_true == 1), (y_pred == 1))] = 1
print "true pos \n",true_pos
true_pos = np.sum(true_pos)
print true_pos

# False Positives
# you predict +1 but it's actually -1
false_positives = np.zeros(scores.shape)
false_positives[np.logical_and((y_true == -1), (y_pred == 1))] = 1
print "false pos \n", false_positives
false_positives = np.sum(false_positives)
print false_positives

# False Negatives
# you predict -1 but it's actually 1
false_negatives = np.zeros(scores.shape)
false_negatives[np.logical_and((y_true == 1), (y_pred == -1))] = 1
print "false neg \n", false_negatives
false_negatives = np.sum(false_negatives)
print false_negatives

precision = float(true_pos) / (true_pos + false_positives)
recall = float(true_pos) / (true_pos + false_negatives)
F1 = 2. * precision * recall / (precision + recall)

print precision, recall, F1


# print y_true==y_pred
# tp[y_true == y_pred] = 1
# print "\n", tp

