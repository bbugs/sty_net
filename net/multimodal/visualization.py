"""


"""

# TODO: Make this into a module that takes in queries and visualizes either regions or words

import numpy as np
from cs231n.multimodal.evaluation import metrics

class Evaluator(object):
    """

    """

    def __init__(self):
        self.model = ''
        return

    def get_ranks(self, scores, y_true):
        y_pred = np.argsort(-scores, axis=1)



    def check_avg_recall_at_k(self, scores, y_true, num_samples, k):
        """(np array, list of lists, int, int)

        Input:
            - scores[i,j]: score of instance i for class j

        """
        N = scores.shape[0]

        # Get the top k predictions for each row


        y_pred = np.argsort(-scores, axis=1)[:, 0:k]



        return


if __name__ == '__main__':


    e = Evaluator()