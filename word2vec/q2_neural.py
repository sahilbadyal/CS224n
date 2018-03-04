#!/usr/bin/env python

import numpy as np
import random

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive


def forward_backward_prop(data, labels, params, dimensions):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Arguments:
    data -- M x Dx matrix, where each row is a training example.
    labels -- M x Dy matrix, where each row is a one-hot vector.
    params -- Model parameters, these are unpacked for you.
    dimensions -- A tuple of input dimension, number of hidden units
                  and output dimension
    """
    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])

    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))

   # print np.shape(W1)
    #print np.shape(b1)
   # print np.shape(W2)
    #print np.shape(b2)

    ### YOUR CODE HERE: forward propagation
    hi = np.add(np.dot(data,W1),b1) #m*h
    
    ho = sigmoid(hi) #m*h

    fi = np.add(np.dot(ho,W2),b2) #m*dy

    fo = softmax(fi) #m*dy

    ### END YOUR CODE
    cost =  np.negative(np.sum(np.log(fo) * labels))
    

    ### YOUR CODE HERE: backward propagation

    value1 = fo - labels # m*dy
    sigmoidGrad = sigmoid_grad(ho) #m*h
    value2 = np.dot(value1,W2.T) # m * Dy X Dy * H = m *H
    value3 = np.multiply(value2,sigmoidGrad) #m*H

    gradb2 = np.sum(value1,axis=0) #(1 * Dy)
    
    gradW2 = np.dot(ho.T,value1)  # h*m X m*dy
 
    gradb1 = np.sum(value3,axis=0) #1 * H

    gradW1 = np.dot(data.T,value3) # dx * m X m * H

    ### END YOUR CODE

    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
                           gradW2.flatten(), gradb2.flatten()))

    return cost, grad


def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])   # each row will be a datum
    labels = np.zeros((N, dimensions[2]))
    for i in xrange(N):
        labels[i, random.randint(0,dimensions[2]-1)] = 1

    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params:
        forward_backward_prop(data, labels, params, dimensions), params)


def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    ### YOUR CODE HERE
    ### END YOUR CODE


if __name__ == "__main__":
    sanity_check()
    your_sanity_checks()
