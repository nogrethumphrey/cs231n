from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Width/height of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype
        C,H,W = input_dim
        
        ##after convolution layer,the dimention remains the same
        W1 = np.random.normal(0,weight_scale,(num_filters, C,filter_size,filter_size))
        b1 = np.zeros(num_filters)

        ##after pool layer,the dimention will decrease (W-pool_size)/pool_size+1
        pool_output_H = int((H-2)/2+1)
        pool_output_w = int((W-2)/2+1)

        ###hidden layer,stretching the output of max pool into one vector
        W2 = np.random.normal(0,weight_scale,(num_filters*pool_output_H*pool_output_w,hidden_dim))
        b2 = np.zeros(hidden_dim)

        W3 = np.random.normal(0,weight_scale,(hidden_dim,num_classes))
        b3 = np.zeros(num_classes)
        
        self.params['W1'] = W1
        self.params['b1'] = b1
        self.params['W2'] = W2
        self.params['b2'] = b2
        self.params['W3'] = W3
        self.params['b3'] = b3

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        loss, grads = 0, {}

        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        #import pdb; pdb.set_trace()

        conv_pool_out, conv_pool_cache = conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
        hidden_out,hidden_cache = affine_relu_forward(conv_pool_out,W2,b2)###dimention will change
        scores,last_affine_layer_cache = affine_forward(hidden_out,W3,b3)

        loss,loss_to_last_affine_gradient = softmax_loss(scores,y)
        loss += 0.5*self.reg*(np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2']))+np.sum(np.square(self.params['W3'])))
        #if np.sum(np.square(self.params['W1'])) >= 1000:
        #    import pdb; pdb.set_trace()
        
        loss_to_hidden_layer_out_gradient,dW3,db3 = affine_backward(loss_to_last_affine_gradient,last_affine_layer_cache)
        loss_to_pool_out_gradient,dW2,db2= affine_relu_backward(loss_to_hidden_layer_out_gradient,hidden_cache)##affine backward dimention is one flattened vector.
        loss_to_pool_out_gradient = loss_to_pool_out_gradient.reshape(conv_pool_out.shape)
        loss_to_input_gradient,dW1,db1= conv_relu_pool_backward(loss_to_pool_out_gradient,conv_pool_cache)
       
        dW3 += self.reg * self.params['W3'] ##
        dW2 += self.reg * self.params['W2'] ##
        dW1 += self.reg * self.params['W1'] ##

        grads['W3'] = dW3
        grads['b3'] = db3
        
        grads['W2'] = dW2
        grads['b2'] = db2

        grads['W1'] = dW1
        grads['b1'] = db1
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
