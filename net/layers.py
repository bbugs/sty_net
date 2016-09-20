import numpy as np
from scipy.sparse import coo_matrix


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    #############################################################################
    # TODO: Implement the affine forward pass. Store the result in out. You     #
    # will need to reshape the input into rows.                                 #
    #############################################################################
    N = x.shape[0]
    D = np.prod(x.shape[1:])
    x_2d = np.reshape(np.copy(x), (N, D))  # reshape x to a 2D array
    out = np.dot(x_2d, w) + b

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache  # unpack cache values

    #############################################################################
    # TODO: Implement the affine backward pass.                                 #
    #############################################################################
    # N = x.shape[0]
    # D = np.prod(x.shape[1:])
    # x_2d = x.reshape(N, D)  # note numpy arrays are mutable, so I'm not sure what happens here
    # dw = np.dot(x_2d.T, dout)  # x_2d.T is DxN and dout is NxM. Thus dw is DxM
    dw = np.dot(np.copy(x).reshape(x.shape[0], np.prod(x.shape[1:])).T, dout)
    db = np.sum(dout, axis=0)

    dx = np.dot(dout, w.T).reshape(x.shape)  # N,d1,d2,...,d_d_k
    # dout is N,M. w.T is M,D. Thus dx is N,D. Then reshape D to d1,d2,..,d_k

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    cache = np.copy(x)
    #############################################################################
    # TODO: Implement the ReLU forward pass.                                    #
    #############################################################################
    out = np.copy(x)
    out[x < 0] = 0

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """

    #############################################################################
    # TODO: Implement the ReLU backward pass.                                   #
    #############################################################################
    dx, x = None, cache
    # local_grad =
    # local_grad[x < 0] = 0

    dx = np.copy(dout)
    dx[x <= 0] = 0

    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the mean
    and variance of each feature, and these averages are used to normalize data
    at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7 implementation
    of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #############################################################################
        # TODO: Implement the training-time forward pass for batch normalization.   #
        # Use minibatch statistics to compute the mean and variance, use these      #
        # statistics to normalize the incoming data, and scale and shift the        #
        # normalized data using gamma and beta.                                     #
        #                                                                           #
        # You should store the output in the variable out. Any intermediates that   #
        # you need for the backward pass should be stored in the cache variable.    #
        #                                                                           #
        # You should also use your computed sample mean and variance together with  #
        # the momentum variable to update the running mean and running variance,    #
        # storing your result in the running_mean and running_var variables.        #
        #############################################################################
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    elif mode == 'test':
        #############################################################################
        # TODO: Implement the test-time forward pass for batch normalization. Use   #
        # the running mean and variance to normalize the incoming data, then scale  #
        # and shift the normalized data using gamma and beta. Store the result in   #
        # the out variable.                                                         #
        #############################################################################
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    #############################################################################
    # TODO: Implement the backward pass for batch normalization. Store the      #
    # results in the dx, dgamma, and dbeta variables.                           #
    #                                                                           #
    # After computing the gradient with respect to the centered inputs, you     #
    # should be able to compute gradients with respect to the inputs in a       #
    # single statement; our implementation fits on a single 80-character line.  #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not in
        real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: A tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                            #
        ###########################################################################
        pass
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        ###########################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.       #
        ###########################################################################
        pass
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        ###########################################################################
        # TODO: Implement the training phase backward pass for inverted dropout.  #
        ###########################################################################
        pass
        ###########################################################################
        #                            END OF YOUR CODE                             #
        ###########################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and width
    W. We convolve each input with F different filters, where each filter spans
    all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the convolutional forward pass.                           #
    # Hint: you can use the function np.pad for padding.                        #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    #############################################################################
    # TODO: Implement the convolutional backward pass.                          #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    #############################################################################
    # TODO: Implement the max pooling forward pass                              #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #############################################################################
    # TODO: Implement the max pooling backward pass                             #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    #############################################################################
    # TODO: Implement the forward pass for spatial batch normalization.         #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    #############################################################################
    # TODO: Implement the backward pass for spatial batch normalization.        #
    #                                                                           #
    # HINT: You can implement spatial batch normalization using the vanilla     #
    # version of batch normalization defined above. Your implementation should  #
    # be very short; ours is less than five lines.                              #
    #############################################################################
    pass
    #############################################################################
    #                             END OF YOUR CODE                              #
    #############################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / float(N)
    num_pos = np.sum(margins > 0, axis=1)  # number of positives
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= float(N)
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    probs = np.exp(x - np.max(x, axis=1, keepdims=True))
    probs /= np.sum(probs, axis=1, keepdims=True)
    N = x.shape[0]
    loss = -np.sum(np.log(probs[np.arange(N), y])) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def svm_struct_loss(x, y, delta=1.0, avg=True):
    """(np array, np array, float, bool) -> float, np array

    author: susana

    Computes the loss and gradient for svm classification, where the
    scores of the diagonal are considered correct classes and the rest of
    the elements are wrong classes.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth class
      for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C
      Note: There is an implicit assumtion that y = np.array(np.arange(N)). This
      makes that the element of the diagonal are the correct classes.

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x

    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins_rows = np.maximum(0, x - correct_class_scores[:, np.newaxis] + delta)
    margins_cols = np.maximum(0, x - correct_class_scores[np.newaxis, :] + delta)

    margins_rows[np.arange(N), y] = 0
    margins_cols[np.arange(N), y] = 0

    margins = margins_rows + margins_cols
    margins[np.arange(N), y] = 0

    loss = np.sum(margins)

    num_pos = np.sum(margins > 0, axis=1)  # number of positives for each row
    num_pos += np.sum(margins > 0, axis=0)  # number of positives for each column
    dx = np.zeros_like(x)
    dx[margins_rows > 0] += 1
    dx[margins_cols > 0] += 1

    dx[np.arange(N), y] -= num_pos  # elements of the diagonal of dx
    if avg:  # want the average loss and dx?
        loss /= float(N)
        dx /= float(N)

    return loss, dx


def mult_forward(x, w):
    """
    Computes the forward pass for a multiplication layer layer.

    Inputs:
    - x: A numpy array containing input data, of shape (d_1, d_2)
    - w: A numpy array of weights, of shape (d_2, d_3)

    Returns a tuple of:
    - out: output, of shape (d_1, d_3)
    - cache: (x, w)
    """
    #############################################################################
    # Implement the affine forward pass. Store the result in out.               #
    #############################################################################
    out = np.dot(x, w)  # (d_1, d_3)

    cache = (x, w)
    return out, cache


def mult_backward(dout, cache):
    """
    Computes the backward pass for an multiplication layer.

    Inputs:
    - dout: Upstream derivative, of shape (d_1, d_3)
    - cache: Tuple of:
      - x: Input data, of shape (d_1, d_2)
      - w: Weights, of shape (d_2, d_3)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (d_1, d_2)
    - dw: Gradient with respect to w, of shape (d_2, d_3)
    """
    x, w = cache  # unpack cache values

    #############################################################################
    # Implement the affine backward pass.                                 #
    #############################################################################

    dx = np.dot(dout, w.T)  # dx: (d_1, d_2)

    dw = np.dot(x.T, dout)  # dw: (d_2, d_3)

    return dx, dw


def get_normalization_weights(y):
    """ np.array -> np.array

    Given a matrix y of size (d1, d2), where each element y[i,j] is either +1 or -1.
     compute a matrix W as follows

    Example:
         y = np.array(
                [[1, 1, -1, -1],
                 [1, 1, -1, -1],
                 [-1, -1, 1, 1],
                 [-1, -1, 1, 1],
                 [-1, -1, 1, 1]])

        W = np.array(
            [[0.5 0.5 0.5 0.5],
            [0.5 0.5 0.5 0.5],
            [0.33 0.33 0.33 0.33],
            [0.33 0.33 0.33 0.33],
            [0.33 0.33 0.33 0.33]]
        )

    """
    # Create indicator variable MEQ that indicates correct pairs of region-word
    MEQ = np.zeros(y.shape, dtype=int)
    MEQ[y == 1] = 1

    ypos = np.zeros(y.shape)
    yneg = np.zeros(y.shape)

    ypos[y == 1] = 1
    yneg[y == -1] = 1

    tmp_pos = ypos / np.sum(ypos, axis=0)
    tmp_neg = yneg / np.sum(yneg, axis=0)

    # substitute nan's by 0.
    tmp_pos[np.isnan(tmp_pos)] = 0
    tmp_neg[np.isnan(tmp_neg)] = 0

    W = np.multiply(tmp_pos, MEQ) + np.multiply(tmp_neg, np.logical_not(MEQ))

    return W


def perform_mil(x, y):
    """
    Inputs:
    - x: Input data, of shape (n_region, n_words) where x[i, j] is the score for the ith image
    region and the jth word.
    - y: Vector of labels, of shape (n_region, n_words) where y[i,j] indicates whether the
      img region i and the jth word occurred together (y[i,j] = 1), or not (y[i,j]=-1)

    Returns:
        A new y array, where we assign y = sign(x) only for the correct pairs
    """
    y_copy = np.copy(y)

    MEQ = np.zeros(y.shape, dtype=int)
    MEQ[y == 1] = 1
    MEQ[y == -1] = 0

    fpos = np.multiply(x, MEQ) - 9999 * np.logical_not(MEQ)
    Ypos = np.sign(fpos)

    # ixbad contains the indices of the columns where no element (of the correct region-word pair) is equal 1.
    ixbad = np.argwhere(np.logical_not(np.any(Ypos == 1, axis=0))).ravel()

    if ixbad.size != 0:  # check if not empty
        # get the row id where the bad index happened
        fmaxi = np.argmax(fpos[:, ixbad], axis=0).ravel()
        data = np.atleast_1d(2 * np.squeeze(np.ones(ixbad.shape)))
        if len(data.shape) == 0:
            data = data.reshape(1)
        Ypos = Ypos + coo_matrix((data, (fmaxi, ixbad)), shape=y_copy.shape)  # flip from -1 to 1: add 2

    # replace the values of the correct region-word pairs with the sign(region' * word)
    # y_copy[MEQ==1] = Ypos[MEQ==1] this doesn't work out in numpy, or this y_copy[MEQ] = Ypos[MEQ==1]

    y_new = np.multiply(y_copy, np.logical_not(MEQ)) + np.multiply(MEQ, Ypos)

    return y_new


def svm_two_classes(x, y, delta=1, do_mil=False, normalize=True):
    """(np array, np array, int) -> float, np array

    author: susana

    Computes the loss and gradient for softmax classification, where the
    scores of the diagonal are considered correct classes and the rest of
    the elements are wrong classes.
    Each element x[i,j] is considered 1 datapoint. And the element y[i,j] indicates
    whether x[i,j] belongs to class +1 or -1.

    Inputs:
    - x: Input data, of shape (n_region, n_words) where x[i, j] is the score for the ith image
    region and the jth word.
    - y: array of labels, of shape (n_region, n_words) where y[i,j] indicates whether the
      img region i and the jth word occurred together (y[i,j] = 1), or not (y[i,j]=-1)
    - delta: how much margin between data points of the two classes (lmargin in matlab code).

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x

    """
    y_copy = np.copy(y)
    Y = np.copy(y)
    dx = np.zeros(x.shape)
    norm_weights = np.ones(y_copy.shape)

    if do_mil:
        Y = perform_mil(x, y_copy)

    margins = np.maximum(0, delta - np.multiply(Y, x))

    if normalize:
        norm_weights = get_normalization_weights(Y)

    weighted_margins = np.multiply(norm_weights, margins)
    loss = np.sum(weighted_margins)

    # dx = dx - np.multiply(margins > 0, Y)  # without norm_weights

    dx = dx - np.multiply(margins > 0, np.multiply(Y, norm_weights))  # with norm weights

    return loss, dx


def sigmoid_forward(x):
    # TODO
    out = 1.0/ (1.0 + np.exp(-x))
    return out, np.copy(out)


def sigmoid_backward(dout, cache):
    # TODO

    dx = cache * (1. - cache) * dout

    return dx


def sigmoid_cross_entropy_loss(z, y):
    """
    Inputs:
    - z: Input data, of shape (N, C) where z[i, j] is the score for the jth class
      for the ith input.
    - y: array of labels, the same size as x. y[i,j] indicates whether the jth
      class is correct for the ith element.
      Each element in z can have more than one correct class.

    Returns a tuple of:
    - loss: scalar giving the loss
    - dz: gradient of the loss with respect to z

    """

    num_elements = np.prod(z.shape)  # N*C

    y_hat = 1.0 / (1.0 + np.exp(-z))

    loss = -np.sum(y * np.log(y_hat) + (1.0 - y) * np.log(1.0 - y_hat)) / num_elements

    dz = (y_hat - y) / num_elements

    return loss, dz


def _global_score_one_pair_forward(sim_img_i_sent_j, smooth_num, **kwargs):
    """
    Forward pass of global_score gate for ONE image-sentence pair
    All image-sentence pairs have a global score,
    average or max to get the global score or region and sentence

    Inputs:

    - sim_img_i_sent_j: np array of size (n_regions_in_img_i, n_words_in_sentence_j) containing local scores
                        sim_img_i_sent_j[i,j] contains the similarity
                        between the ith region and the jth word for ONE given
                        image-sentence pair (correct or incorrect does not matter.

    - smooth_num:       scalar smoothing constant, i.e., 'n' in Eq. 6 of NIPS paper.

    - kwargs:           global_method is either 'sum' or 'maxaccum' (indicates how to compute
                            the global score, either with a sum or with a max)
                        thrglobalscore is either True or False (indicates whether
                            to threshold global scores at zero)


    Returns:
         -s: a scalar with the global score of the image-sentence pair
         -nnorm: a normalization scalar
         - img_region_index_with_max for backprop in case of maxaccum

    """

    # local_to_global_score_one_pair
    # unpack keyword arguments
    thrglobalscore = kwargs['thrglobalscore']
    global_method = kwargs['global_method']
    img_region_index_with_max = np.array([])

    num_regions, num_words = sim_img_i_sent_j.shape

    if thrglobalscore:
        sim_img_i_sent_j[sim_img_i_sent_j < 0] = 0  # threshold at zero

    if global_method == 'sum':
        global_score = np.sum(sim_img_i_sent_j) # score of image-sentence

    elif global_method == 'maxaccum':
        # for each word, find the closest (in dot product) image region
        max_sim_for_each_word = np.max(sim_img_i_sent_j, axis=0)  # the max value of sim for each word (1, num_words)
        img_region_index_with_max = np.argmax(sim_img_i_sent_j, axis=0)  # recall this for backprop (1, num_words)

        global_score = np.sum(max_sim_for_each_word)  # score of image-sentence
    else:
        raise ValueError("global method must be either sum or maxaccum")

    nnorm = float(num_words + smooth_num)
    global_score /= nnorm

    cache = {}
    cache['nnorm'] = nnorm
    cache['img_region_index_with_max'] = img_region_index_with_max
    cache['n_regions_n_words'] = sim_img_i_sent_j.shape
    # cache['sim_img_i_sent_j'] = sim_img_i_sent_j

    return global_score, cache


def global_scores_forward(sim_region_word, N, region2pair_id, word2pair_id, smooth_num=5, **kwargs):
    """
    Forward pass of global_scores gate

    Inputs:
    - sim_region_word:  np array of size (n_regions_in_batch, n_words_in_batch) containing local scores
    - N:                number of correct image-sentence pairs in batch
    - region2pair_id:   np array of size (1, n_regions_n_batch). The index is the region id
                            and the value is the pair id that the region belongs to.
    - word2pair_id:     np array of size (1, n_words_in_batch). The index is the region id
                            and the value is the pair id that the word belongs to.

    Returns:
    - img_sent_global_score:  np array of size (N,N) containing the global
                              scores for each image-sentence pair in batch (correct and incorrect ones)
    """

    img_sent_score_global = np.zeros((N, N))
    SGN = np.zeros((N, N))
    img_region_with_max = {}

    for i in range(N):
        for j in range(N):
            # sim_img_i_sent_j = local_scores_all[region2pair_id == i, word2pair_id == j]

            # sim_img_i_sent_j = local_scores_all[np.where(region2pair_id==i)[0], np.where(word2pair_id == j)[0]]

            tmp = sim_region_word[np.where(region2pair_id == i)][:, np.where(word2pair_id == j)]
            sim_img_i_sent_j = np.squeeze(tmp)

            s, info = _global_score_one_pair_forward(sim_img_i_sent_j, smooth_num, **kwargs)

            if kwargs['global_method'] == 'maxaccum':
                img_region_with_max[(i, j)] = info['img_region_index_with_max']

            img_sent_score_global[i, j] = s
            SGN[i, j] = info['nnorm']

    return img_sent_score_global, SGN, img_region_with_max


def _global_score_one_pair_backward(dout, nnorm, sim_img_i_sent_j, img_region_index_with_max, **kwargs):
    """
    Backward pass of global_scores gate for one pair

    Inputs:
    - dout:             upstream gradient of size (1,1), i.e., a scalar
    - sim_img_i_sent_j: np array of size (n_regions_in_img_i, n_words_in_sentence_j) containing local scores
    - nnorm:            scalar containing normalizing constants previously
                        computed in the forward pass.
    - kwargs:           global_method is either 'sum' or 'maxaccum'

    Returns:
    - d_local_scores:   gradient wrt local scores (sim_img_i_sent_j), same size as sim_img_i_sent_j.
    """

    thrglobalscore = kwargs['thrglobalscore']
    global_method = kwargs['global_method']

    n_regions, n_words = sim_img_i_sent_j.shape

    if global_method == 'sum':
        d_local_scores = np.ones((n_regions, n_words)) * dout / nnorm

    elif global_method == 'maxaccum':
        d_local_scores = np.zeros((n_regions, n_words))
        d_local_scores[img_region_index_with_max, np.arange(n_words)] = dout / nnorm

    else:
        raise ValueError("global method must be sum or maxaccum")

    if thrglobalscore:
        d_local_scores[sim_img_i_sent_j < 0] = 0

    return d_local_scores


def global_scores_backward(dout, N, sim_region_word,
                           region2pair_id, word2pair_id, nnorm, img_region_index_with_max, **kwargs):
    """
    Backward pass of global_scores gate
    Inputs:
    - dout:             upstream gradient of size (N,N)
    - sim_region_word:  np array of size (n_regions_in_batch, n_words_in_batch) containing local scores
    - region2pair_id:   np array of size (1, n_regions_n_batch). The index is the region id
                            and the value is the pair id that the region belongs to.
    - word2pair_id:     np array of size (1, n_words_in_batch). The index is the region id
                            and the value is the pair id that the word belongs to.
    - nnorm:            np array of size (N,N) containing normalizing constants previously
                            computed in the forward pass.
    - kwargs:           global_method is either 'sum' or 'maxaccum' (indicates how to compute
                            the global score, either with a sum or with a max)
                        thrglobalscore is either True or False (indicates whether
                            to threshold global scores at zero)

    Returns:
    - d_local_scores:   gradient wrt local scores (sim_region_word), same size as sim_region_word.
    """
    # unpack keyword arguments
    global_method = kwargs['global_method']

    d_local_scores = np.zeros(sim_region_word.shape)

    for i in range(N):
        for j in range(N):
            # slice out the local scores that correspond to all regions of image i and all words of sentence j
            tmp = sim_region_word[np.where(region2pair_id == i)][:, np.where(word2pair_id == j)]
            sim_img_i_sent_j = np.squeeze(tmp)

            # indicator array of region-word pairs corresponding to the ith image and the jth sentence
            MEQ = np.outer(region2pair_id == i, word2pair_id == j)

            if global_method == 'sum':
                # get gradient with respect to the local scores of image i and sentence j
                dd = _global_score_one_pair_backward(dout[i, j], nnorm[i, j], sim_img_i_sent_j,
                                                     img_region_index_with_max, **kwargs)
            elif global_method == 'maxaccum':
                dd = _global_score_one_pair_backward(dout[i, j], nnorm[i, j], sim_img_i_sent_j,
                                                     img_region_index_with_max[(i, j)], **kwargs)
            else:
                raise ValueError("only sum and maxaccum are supported as methods to compute global scores")

            d_local_scores[MEQ] = dd.ravel()

    return d_local_scores



# Sus:
if __name__ == "__main__":
    #     from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
    #     x = np.random.randn(10, 2, 3)
    #     w = np.random.randn(6, 5)
    #     b = np.random.randn(5)
    #     dout = np.random.randn(10, 5)
    #
    #     dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
    #     dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
    #     db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)
    #
    #     _, cache = affine_forward(x, w, b)
    #     dx, dw, db = affine_backward(dout, cache)
    #
    #     pass

    # scores
    pass
