import numpy as np
from net.multimodal.data_provider.json_data import JsonFile
import itertools
import logging
import pickle
import os
import random
import json


def init_random_weights(n_rows, n_cols=None):
    """
    This function aims to address the fact that numpy.random does not
    provide a random output at runtime
    Args:
        n_rows:
        n_cols:

    Returns:

    """

    if n_cols is None:
        w = np.zeros((n_rows))
        for i in range(n_rows):
            w[i] = random.gauss(mu=0.0, sigma=1)
        return w
    else:
        w = np.zeros((n_rows, n_cols))
        for i in range(n_rows):
            for j in range(n_cols):
                w[i, j] = random.gauss(mu=0.0, sigma=1)
        return w


def write_report_for_exp_id(new_report_fname, new_report, exp_config, current_val_f1):
    report_path = exp_config['checkpoint_path']
    old_reports = [f.replace('.pkl', '') for
                   f in os.listdir(report_path) if f.startswith("report_valf1_")]

    to_consider_for_delete = []

    for old_report in old_reports:
        config = old_report.split("_")
        old_exp_id = int(config[4])

        if old_exp_id == exp_config['id']:
            to_consider_for_delete.append(old_report)

    # if there are no potentials to delete, just save the report and return
    if len(to_consider_for_delete) == 0:
        # save and return
        with open(report_path + new_report_fname, "wb") as f:
            pickle.dump(new_report, f)

        logging.info("id_{} saved report to {}".format(exp_config['id'], new_report_fname))
        return

    # see whether to keep old report or new one
    for old_report in to_consider_for_delete:
        config = old_report.split("_")
        old_val_f1 = float(config[2])

        if current_val_f1 >= old_val_f1:  # if results are just as good, save because I'd like to see
            # if the epochs advanced or not.
            # delete old report and save new one and return
            os.remove(report_path + old_report + '.pkl')  # delete
            with open(report_path + new_report_fname, "wb") as f:  # save
                pickle.dump(new_report, f)
            logging.info("id_{} replaced report {} by {}".
                         format(exp_config['id'], old_report, new_report_fname))
            return

        elif old_val_f1 > current_val_f1:
            # no need to save anything.
            return

        else:
            msg = "THIS SHOULD NOT HAPPEN... SOMETHING WENT WRONG"
            logging.info(msg)
            print msg
            return


def write_report(new_report_fname, new_report, exp_config, current_val_f1):

    # see if there is another file of the same consition as in ex_config

    # get old reports with same condition
    report_path = exp_config['checkpoint_path']
    old_reports = [f.replace('.pkl', '') for
                   f in os.listdir(report_path) if f.startswith("report_valf1_")]

    to_consider_for_delete = []
    for old_report in old_reports:
        config = old_report.split("_")

        old_hidden_dim = int(config[6])
        old_use_local = float(config[8])
        old_use_global = float(config[10])
        old_use_associat = float(config[12])

        if (old_use_local == exp_config['use_local'] and
           old_use_global == exp_config['use_global'] and
           old_use_associat == exp_config['use_associat'] and
           old_hidden_dim == exp_config['hidden_dim']):
            to_consider_for_delete.append(old_report)

    # if there are no potentials to delete, just save the report and return
    if len(to_consider_for_delete) == 0:
        # save and return
        with open(report_path + new_report_fname, "wb") as f:
            pickle.dump(new_report, f)

        logging.info("id_{} saved report to {}".format(exp_config['id'], new_report_fname))
        return

    # see whether to keep old report or new one
    for old_report in to_consider_for_delete:
        config = old_report.split("_")
        old_val_f1 = float(config[2])

        if current_val_f1 > old_val_f1:
            # delete old report and save new one and return
            os.remove(report_path + old_report + '.pkl')  # delete
            with open(report_path + new_report_fname, "wb") as f:  # save
                pickle.dump(new_report, f)
            logging.info("id_{} replaced report {} by {}".
                         format(exp_config['id'], old_report, new_report_fname))
            return

        elif old_val_f1 >= current_val_f1:
            # no need to save anything. old results are better or just as good
            return

        else:
            msg = "THIS SHOULD NOT HAPPEN... SOMETHING WENT WRONG"
            logging.info(msg)
            print msg
            return

def check_img_ids(json_fname, imgid2region_indices):
    """(str, dict) ->

    imgid2region_indices is a dict whose

    """
    json_file = JsonFile(json_fname, num_items=-1)

    assert sorted(imgid2region_indices.keys()) == json_file.get_img_ids(), \
           "mismatch between img ids from imgid2region_indices and json file {}".format(json_fname)

    return


def check_num_regions(cnn_fname, imgid2region_indices, verbose=False):
    """(str, dict) ->
    compare the number of regions from cnn_fname and the number
    of regions from imgid2region_indices

    """
    num_regions = 0
    for img_id in imgid2region_indices:
        num_regions += len(imgid2region_indices[img_id])

    num_regions_cnn_array = get_num_lines_from_file(cnn_fname)

    if verbose:
        print "num regions from imgid2region_indices: ", num_regions
        print "num regions from cnn array: ", num_regions_cnn_array

    assert num_regions == num_regions_cnn_array, \
           "mismatch between num_regions and num_lines in cnn file {}".format(cnn_fname)
    return

def mk_toy_img_id2region_indices(json_fname, cnn_fname, num_regions_per_img, subset_num_items=-1):

    json_file = JsonFile(json_fname, num_items=subset_num_items)

    img_ids = json_file.get_img_ids()

    img_id2region_indices = {}

    region_index = 0
    for img_id in img_ids:
        img_id2region_indices[img_id] = []
        for i in range(num_regions_per_img):
            img_id2region_indices[img_id].append(region_index)
            region_index += 1

    # verify consistency between img ids in json file and the img_id2region_indices
    check_img_ids(json_fname, img_id2region_indices)
    # verify consistency between the cnn file and the number of regions
    check_num_regions(cnn_fname, img_id2region_indices)

    return img_id2region_indices


def mk_cnn_region_index2img_id(img_id2region_indices):
    cnn_region_index2img_id = {}
    for img_id in img_id2region_indices:
        region_indices = img_id2region_indices[img_id]
        for region_index in region_indices:
            cnn_region_index2img_id[region_index] = img_id
    return cnn_region_index2img_id


def get_num_lines_from_file(fname):
    counts = itertools.count()
    with open(fname) as f:
        for _ in f: counts.next()
    return counts.next()


def y2pair_id(y, N):
    """

    """

    region2pair_id = np.zeros(yy.shape[0])

    word2pair_id = np.zeros(yy.shape[1])

    i = 0
    k_region = 0
    k_word = 0
    col_counter = 0
    row_counter = 0
    while i < N:

        num_regions = len(np.where(y[:, col_counter] == 1)[0])
        num_words = len(np.where(y[row_counter, :] == 1)[0])

        region2pair_id[k_region: k_region + num_regions] = i
        word2pair_id[k_word: k_word + num_words] = i

        i += 1
        k_region += num_regions
        k_word += num_words

        row_counter += num_regions
        col_counter += num_words

    return region2pair_id, word2pair_id


def pair_id2y(region2pair_id, word2pair_id):

    N = np.max(region2pair_id)
    assert N == np.max(word2pair_id)

    n_regions = region2pair_id.shape[0]
    n_words = word2pair_id.shape[0]
    y = -np.ones((n_regions, n_words))

    for i in range(N + 1):
        MEQ = np.outer(region2pair_id == i, word2pair_id == i)
        y[MEQ] = 1

    return y


if __name__ == '__main__':

    yy = np.array([[1, 1, -1, -1, -1, -1, -1],
                  [1, 1, -1, -1, -1, -1, -1],
                  [-1, -1, 1, 1, -1, -1, -1],
                  [-1, -1, 1, 1, -1, -1, -1],
                  [-1, -1, 1, 1, -1, -1, -1],
                  [-1, -1, -1, -1, 1, 1, 1],
                  [-1, -1, -1, -1, 1, 1, 1],
                  [-1, -1, -1, -1, 1, 1, 1]], dtype=np.float)

    r2p = np.array([0, 0, 1, 1, 1, 2, 2, 2])
    w2p = np.array([0, 0, 1, 1, 2, 2, 2])

    assert np.allclose(pair_id2y(r2p, w2p), yy)
    #
    # print y2pair_id(yy, N=3)

    from net.multimodal.data_provider.data_tests import test_data_config

    json_ffname = test_data_config.exp_config['json_path_test']
    cnn_ffname = test_data_config.exp_config['cnn_regions_path_test']
    imgid2regionind = mk_toy_img_id2region_indices(json_fname=json_ffname,
                                                   cnn_fname=cnn_ffname,
                                                   num_regions_per_img=5,
                                                   subset_num_items=-1)
    correct = {}
    correct[6] = [0,1,2,3,4]
    correct[80] = [5,6,7,8,9]
    correct[147] = [10, 11, 12, 13, 14]
    print imgid2regionind  # {80: [5, 6, 7, 8, 9], 147: [10, 11, 12, 13, 14], 6: [0, 1, 2, 3, 4]}
    # assert imgid2regionind == correct

    ffname = test_data_config.exp_config['cnn_regions_path_test']
    print get_num_lines_from_file(ffname)


