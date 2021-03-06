from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        ############################################################################
        self.params['W1'] = np.random.normal(0, weight_scale, [input_dim, hidden_dim])
        self.params['b1'] = np.zeros([hidden_dim])
        self.params['W2'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
        self.params['b2'] = np.zeros([num_classes])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        #####Here the cache is really strange
        layer_one_out, layer_one_cache = affine_relu_forward(X,self.params['W1'],self.params['b1'])
        layer_two_out,layer_two_cache= affine_forward(layer_one_out,self.params['W2'],self.params['b2'])
        scores = layer_two_out
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss,loss_to_layer_two_output_gradient = softmax_loss(scores,y)
        loss += 0.5*self.reg*(np.sum(np.square(self.params['W1'])) + np.sum(np.square(self.params['W2'])))
        if(np.sum(np.square(self.params['W1']) >= 1000)):
          import pdb; pdb.set_trace()

        loss_to_layer_one_output_gradient,dW2,db2 = affine_backward(loss_to_layer_two_output_gradient,layer_two_cache)

        dW2 += self.reg * self.params['W2'] ##

        grads['W2'] = dW2
        grads['b2'] = db2

        _, dW1, db1 = affine_relu_backward(loss_to_layer_one_output_gradient,layer_one_cache)
        dW1 += self.reg * self.params['W1']
        grads['W1'] = dW1
        grads['b1'] = db1
        
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch/layer normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=1, normalization=None, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=1 then
          the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
          are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.normalization = normalization
        self.use_dropout = dropout != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)###Plus the softmax layer
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #        self.params['W2'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #        self.params['W2'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
        # beta2, etc. Scale parameters should be initialized to ones and shift     #        self.params['W2'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
        # parameters should be initialized to zeros.                               #        self.params['W2'] = np.random.normal(0, weight_scale, [hidden_dim, num_classes])
        ############################################################################

        ############Initialize the parameters############
        ############Hidden layer are the layers directly after input layer and before the softmax layer###
        length = len(hidden_dims)####number of hidden layer
        for i in range(length):
          layer_str = str(i+1)
          if( i == 0):
            self.params['W'+layer_str] = np.random.normal(0, weight_scale, (input_dim, hidden_dims[i]))
            self.params['b'+layer_str] = np.zeros(hidden_dims[i])
          else:
            self.params['W'+layer_str] = np.random.normal(0, weight_scale, (hidden_dims[i-1], hidden_dims[i]))
            self.params['b'+layer_str] = np.zeros(hidden_dims[i])
      
          if self.normalization=='batchnorm' or self.normalization=='layernorm':
            self.params['gamma'+layer_str] = np.ones(hidden_dims[i])
            self.params['beta'+layer_str] = np.zeros(hidden_dims[i])

        ##the last hidden layer before softmax(This is the last affine layer)
        self.params['W'+str(length+1)] = np.random.normal(0, weight_scale, (hidden_dims[length-1],num_classes))
        self.params['b'+str(length+1)] = np.zeros(num_classes)

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization=='batchnorm':
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]
        if self.normalization=='layernorm':
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.normalization=='batchnorm':
            for bn_param in self.bn_params:
                bn_param['mode'] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################

        #####L-1 affine_relu forward
        #####L-1 affine_relu backward
        #####1 affine forward
        #####1 affine backward
        layer_cache = [None] * (self.num_layers)
        dropout_cache = [None] * (self.num_layers-1)
        dropout_layerout = [None] * (self.num_layers-1)
        layer_out = None

        ##calculate all the forward pass for hidden layers.For batch normalization
        ##We should add batch_norm layer after affine_forward.This means that i have to
        ##use affine_forward,batchnorm_forward,relu_forward in forward pass and
        ##relu_backward,batchnorm_backward,affine_backward in backward pass
        for i in range(self.num_layers-1):
          #import pdb; pdb.set_trace()
          layer_str = str(i+1)
          if(i == 0) :
            if self.normalization=='batchnorm':
              layer_out,layer_cache[i] = affine_relu_batch_forward(X,self.params['W'+layer_str],self.params['b'+layer_str],self.params['gamma'+layer_str],self.params['beta'+layer_str],self.bn_params[i])
            elif self.normalization=='layernorm':
              layer_out,layer_cache[i] = affine_relu_layernorm_forward(X,self.params['W'+layer_str],self.params['b'+layer_str],self.params['gamma'+layer_str],self.params['beta'+layer_str],self.bn_params[i])
            else:
              layer_out,layer_cache[i] = affine_relu_forward(X,self.params['W'+layer_str],self.params['b'+layer_str])
          else:
            if self.normalization=='batchnorm':
              layer_out,layer_cache[i] = affine_relu_batch_forward(layer_out,self.params['W'+layer_str],self.params['b'+layer_str],self.params['gamma'+layer_str],self.params['beta'+layer_str],self.bn_params[i])
            elif self.normalization=='layernorm':
              layer_out,layer_cache[i] = affine_relu_layernorm_forward(layer_out,self.params['W'+layer_str],self.params['b'+layer_str],self.params['gamma'+layer_str],self.params['beta'+layer_str],self.bn_params[i])
            else:
              layer_out,layer_cache[i] = affine_relu_forward(layer_out,self.params['W'+layer_str],self.params['b'+layer_str])
          if self.use_dropout:
            layer_out,dropout_cache[i] = dropout_forward(layer_out,self.dropout_param)

        scores,layer_cache[self.num_layers-1] = affine_forward(layer_out,self.params['W'+str(self.num_layers)],self.params['b'+str(self.num_layers)])
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        
        ############loss should add all the regularization parameters
        loss,loss_to_last_affine_layer_gradient = softmax_loss(scores,y)
        
        ###add all the regularizations to loss
        for i in range(self.num_layers):
          loss += 0.5 * self.reg * np.sum(np.square(self.params['W'+str(i+1)]))

        if(np.sum(np.square(self.params['W1']) >= 1000)):
          pass
        #  import pdb; pdb.set_trace()

        ###Backpropagation should be done reversely
        loss_to_previous_layer_gradient = None
        for i in reversed(range(self.num_layers)):
          layer_str = str(i+1)
          if(i == (self.num_layers -1)):
            loss_to_previous_layer_gradient,grads['W'+layer_str],grads['b'+layer_str] = affine_backward(loss_to_last_affine_layer_gradient,layer_cache[i])
          else:
            if self.use_dropout:
                loss_to_previous_layer_gradient = dropout_backward(loss_to_previous_layer_gradient,dropout_cache[i])
            if self.normalization=='batchnorm':
              loss_to_previous_layer_gradient,grads['W'+layer_str],grads['b'+layer_str],grads['gamma'+layer_str],grads['beta'+layer_str] = affine_relu_batch_backward(loss_to_previous_layer_gradient,layer_cache[i])
            elif self.normalization=='layernorm':
              loss_to_previous_layer_gradient,grads['W'+layer_str],grads['b'+layer_str],grads['gamma'+layer_str],grads['beta'+layer_str] = affine_relu_layernorm_backward(loss_to_previous_layer_gradient,layer_cache[i])
            else:
              loss_to_previous_layer_gradient,grads['W'+layer_str],grads['b'+layer_str] = affine_relu_backward(loss_to_previous_layer_gradient,layer_cache[i])
         
            
          grads['W'+layer_str] += self.reg * self.params['W'+layer_str]
 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
