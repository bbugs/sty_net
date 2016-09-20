import numpy as np


def mk_random_pair_id(n_fragments_per_product, N):

    n_fragments = N * n_fragments_per_product
    fragment2pair_id = np.zeros(n_fragments, dtype=int)
    index = 0
    k = 0
    while index < n_fragments:
        fragment2pair_id[index: index + n_fragments_per_product] = k
        k += 1
        index += n_fragments_per_product
    return fragment2pair_id


def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))


class TestSuite:
    """
    Create a suite of tests similar to unittest
    """

    def __init__(self):
        """
        Creates a test suite object
        """
        self.total_tests = 0
        self.failures = 0

    def run_test(self, computed, expected, message=""):
        """
        Compare computed and expected
        If not equal, print message, computed, expected
        """
        self.total_tests += 1
        if not np.allclose(computed, expected, rtol=1e-05, atol=1e-08):
            print message + " Computed: \n" + str(computed) + \
                  " Expected: \n" + str(expected)
            self.failures += 1

    def report_results(self):
        """
        Report back summary of successes and failures
        from run_test()
        """
        print "Ran " + str(self.total_tests) + " tests. " \
              + str(self.failures) + " failures."
