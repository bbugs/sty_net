
from net.multimodal.data_provider.alignment_data import AlignmentDataV0
from net.multimodal.data_provider.data_tests import test_data_config
import numpy as np

d = test_data_config.exp_config

align_data = AlignmentDataV0(d, split='test', num_items=10)


###################################################
#  Test make_region2pair_id
###################################################
def test_make_region2pair_id():
    img_ids = [2, 3, 4, 65, 45]
    region2pair_id = align_data.make_region2pair_id(img_ids, num_regions_per_img=5)

    correct = np.array([0,0,0,0,0, 1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4])

    assert np.allclose(region2pair_id, correct)

    return


###################################################
#  Test make_word2pair_id
###################################################
def test_make_word2pair_id():
    img_ids = [6, 80, 385]

    word2pair_id = align_data.make_word2pair_id(img_ids, verbose=True)

    print word2pair_id

    correct = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 0., 0., 0., 0., 0., 0., 0., 0.,
                        0., 1., 1., 1., 1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 1., 1., 1., 1., 1.,
                        1., 1., 1., 1., 2., 2., 2., 2., 2.,
                        2., 2.])

    assert np.allclose(word2pair_id, correct)

    return


###################################################
#  Test pair_id2y
###################################################
def test_pair_id2y():
    print "\n\n\n"
    print "testing pair_id2y"
    img_ids = [212, 261, 373, 385]
    region2pair_id = align_data.make_region2pair_id(img_ids, num_regions_per_img=5)
    word2pair_id = align_data.make_word2pair_id(img_ids, verbose=True)

    y = align_data.pair_id2y(region2pair_id, word2pair_id)

    # TODO: make assertions

    print y


###################################################
#  Test make_y_true_txt2img
###################################################
def test_make_y_true_txt2img():
    y_true_txt2img = align_data.make_y_true_txt2img(num_regions_per_img=1)

    print y_true_txt2img
    print y_true_txt2img.shape

    # TODO: make assertions

    return


###################################################
#  Test make_y_true_img2txt
###################################################
# To make y_true_img2txt simply do the transpose of y_true_txt2img

###################################################
#  Test get_img_id2region_index
###################################################

def test_get_img_id2region_index():
    img_id2region_index = align_data.get_img_id2cnn_region_index(num_regions=5)

    print img_id2region_index[6]  # first item
    print img_id2region_index[80]  # second item
    print img_id2region_index[476]  # last item when there are 10 items in test split

    # TODO: make assertions


def main():
    test_make_region2pair_id()
    test_make_word2pair_id()
    test_pair_id2y()
    test_make_y_true_txt2img()
    test_get_img_id2region_index()

if __name__ == "__main__":
    main()