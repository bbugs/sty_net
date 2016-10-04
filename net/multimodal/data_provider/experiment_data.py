from net.multimodal.data_provider.cnn_data import CnnData, check_num_regions
from net.multimodal.data_provider.json_data import JsonFile, check_img_ids
from net.multimodal.data_provider.word2vec_data import Word2VecData


class ExperimentData(object):
    def __init__(self, json_fname, cnn_fname, img_id2cnn_region_indeces,
                 w2v_vocab_fname, w2v_vectors_fname, subset_num_items):
        """
        Inputs:
        json_fname: str
        cnn_fname: str
        batch_size: int
        img_id2cnn_region_indeces:  a dict[img_id] = list of indices of cnn regions

        """
        # check that cnn file and img_id2cnn_region_indeces are consistent
        check_num_regions(cnn_fname, img_id2cnn_region_indeces)
        # check that json file and img_id2cnn_region_indeces are consistent
        check_img_ids(json_fname, img_id2cnn_region_indeces)

        self.json_file = JsonFile(json_fname, num_items=subset_num_items)
        self.cnn_data = CnnData(cnn_fname)
        self.img_id2cnn_region_indeces = img_id2cnn_region_indeces

        self.w2v_data = Word2VecData(w2v_vocab_fname, w2v_vectors_fname)


