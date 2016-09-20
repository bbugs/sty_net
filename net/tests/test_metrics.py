import numpy as np
from cs231n.multimodal.evaluation import metrics

#######################################
# Test case 1
#######################################
# set number of items
n1 = 3
n2 = 4
# create some true values
np.random.seed(42)
y_true = np.random.randint(low=-1, high=1, size=(n1, n2))
y_true[y_true == 0] = 1

# create some predicted values
np.random.seed(22)
y_pred = np.random.randint(low=-1, high=1, size=(n1, n2))
y_pred[y_pred == 0] = 1

print "y_true \n", y_true
print "y_pred \n", y_pred


p, r, f1 = metrics.precision_recall_f1(y_pred=y_pred, y_true=y_true)
print p, r, f1

assert p == 0
assert r == 0


#######################################
# Test case 2
#######################################
# set number of items
n1 = 4
n2 = 4

# create some true values
np.random.seed(42)
y_true = np.random.randint(low=-1, high=1, size=(n1, n2))
y_true[y_true == 0] = 1

# create some predicted values
np.random.seed(122)
y_pred = np.random.randint(low=-1, high=1, size=(n1, n2))
y_pred[y_pred == 0] = 1

print "y_true \n", y_true
print "y_pred \n", y_pred


p, r, f1 = metrics.precision_recall_f1(y_pred=y_pred, y_true=y_true)
print p, r, f1

assert p == 0.2
assert r == 0.25

#######################################
# Test case 3
#######################################
# set number of items
n1 = 3
n2 = 4
# create some true values
np.random.seed(42)
y_true = np.random.randint(low=-1, high=1, size=(n1, n2))
y_true[y_true == 0] = 1

# create some predicted values
np.random.seed(22)
y_pred = np.array([[1, 1, -1, -1],
                  [-1, 1, -1,  1],
                  [-1, 1,  1, -1]])

print "y_true \n", y_true
print "y_pred \n", y_pred


p, r, f1 = metrics.precision_recall_f1(y_pred=y_pred, y_true=y_true)
print p, r, f1

assert p == 0.5
assert r == 1.

