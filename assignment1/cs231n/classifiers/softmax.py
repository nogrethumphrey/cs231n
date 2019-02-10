import numpy as np
from random import shuffle

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  
  ####This is actually the crosss-entropy loss for multiple classes
  num_samples = y.shape[0]
  num_classes = W.shape[1]
  linear_output = X.dot(W)
  total_scores  = np.exp(linear_output - np.max(linear_output,axis=1))
  loss = -np.log(total_scores[np.arange[num_samples],y])
  loss = np.sum(loss) / num_samples +  reg * np.linalg.norm(W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  
  return loss, dW


'''
compute the softmax loss and gradient 

Inputs:
X: NxD sample 
y: array containing the sample number of lables
W: W parameters of this layer.


'''
def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_samples = y.shape[0]
  num_classes = W.shape[1]
  linear_output = X.dot(W)
  
  ###total scores for neural network
  total_scores  = np.exp(linear_output - np.max(linear_output,axis=1))
  total_scores = total_scores / np.sum(total_scores,axis=1)
  
  
  ####cross entropy loss
  loss = -np.log(total_scores[np.arange[num_samples],y])
  loss = np.sum(loss) / num_samples +  0.5 * reg * np.linalg.norm(W)
  
#############################################
## X[sample_index] is actually the output of the previouls layer
## This is actually a short version of back propergation
## It uses previouls_output.dot(backward_gradients) will get the gradients of the parameters.
#############################################
  
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  total_scores[range(num_samples),y] -= 1 ###gradients of loss vs output of second linear layer.
  dW = X.T.dot(total_scores)
  dW + = reg * W

  return loss, dW

'''
This is the forward pass for one single layer.So simple

Inputs:

X:the output of this layer
W:the current parameters of this layer
activition_function:the activition function of this layer.Of course
activition_function could be direct output(without applying any activation function)
'''
def forward(X,W,activition_func):
  return activition_function(X.dot(W))

'''
backward pass of one layer(backward is actually really simple)
Inputs:

X: input of this layer
dW:the gradient of final loss with respect to the output of this layer

return:the gradient of loss function with respect of all the parameters of current layer 
'''
def backward(X,dW):
  return X[:,np.newaxis].dot(dW[np.newaxis,:])