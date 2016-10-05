import linecache

import numpy as np


class CnnData(object):

    def __init__(self, cnn_fname):

        self.fname = cnn_fname
        self.cnn = np.array([])
        self.cnn_dim = 0

        return

    def set_cnn(self):
        self.cnn = np.loadtxt(self.fname, delimiter=',')

    def get_cnn(self):
        if self.cnn.size == 0:
            self.set_cnn()
        return self.cnn

    def get_cnn_from_index(self, index):
        return np.fromstring(linecache.getline(self.fname, index + 1), sep=",")

    def set_cnn_dim(self):
        # read the first line of cnn file and check the dimension
        vec = np.fromstring(linecache.getline(self.fname, 1), sep=",")
        self.cnn_dim = vec.shape[0]

    def get_cnn_dim(self):
        if self.cnn_dim == 0:
            self.set_cnn_dim()
        return self.cnn_dim


    # def set_cnn_full_img_split(self, split='test'):
    #     if split == 'train':
    #         fname = self.d['cnn_full_img_path_train']
    #         self.cnn_full_img_train = np.loadtxt(fname, delimiter=',')
    #     elif split == 'val':
    #         fname = self.d['cnn_full_img_path_val']
    #         self.cnn_full_img_val = np.loadtxt(fname, delimiter=',')
    #     elif split == 'test':
    #         fname = self.d['cnn_full_img_path_test']
    #         self.cnn_full_img_test = np.loadtxt(fname, delimiter=',')
    #     else:
    #         raise ValueError("only train, val and test splits supported")
    #     return
    #
    # def set_cnn_regions_split(self, split):
    #     if split == 'train':
    #         fname = self.d['cnn_regions_path_train']
    #         self.cnn_region_train = np.loadtxt(fname, delimiter=',')
    #     elif split == 'val':
    #         fname = self.d['cnn_regions_path_val']
    #         self.cnn_region_val = np.loadtxt(fname, delimiter=',')
    #     elif split == 'test':
    #         fname = self.d['cnn_regions_path_test']
    #         self.cnn_region_test = np.loadtxt(fname, delimiter=',')
    #     else:
    #         raise ValueError("only train, val and test splits supported")
    #     return
    #
    # def get_cnn_regions_split(self, split):
    #     if split == 'train':
    #         if self.cnn_region_train.size == 0:
    #             self.set_cnn_regions_split(split)
    #         return self.cnn_region_train
    #     elif split == 'val':
    #         if self.cnn_region_val.size == 0:
    #             self.set_cnn_regions_split(split)
    #         return self.cnn_region_val
    #     elif split == 'test':
    #         if self.cnn_region_test.size == 0:
    #             self.set_cnn_regions_split(split)
    #         return self.cnn_region_test
    #     else:
    #         raise ValueError("only train, val and test splits supported")
    #
    # def get_cnn_full_img_split(self, split):
    #     if split == 'train':
    #         if self.cnn_full_img_train.size == 0:
    #             self.set_cnn_full_img_split(split)
    #         return self.cnn_full_img_train
    #     elif split == 'val':
    #         if self.cnn_full_img_val.size == 0:
    #             self.set_cnn_full_img_split(split)
    #         return self.cnn_full_img_val
    #     elif split == 'test':
    #         if self.cnn_full_img_test.size == 0:
    #             self.set_cnn_full_img_split(split)
    #         return self.cnn_full_img_test
    #     else:
    #         raise ValueError("only train, val and test splits supported")
    #
