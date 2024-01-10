# Nakka, Tulasi
# 1001_928_971
# 2023_10_15
# Assignment_02_01

import tensorflow as tf
import numpy as np

def Weight_Matrix(layers, x_train, seed):    
    #Use numpy for weight to initialize weights. Do not use tensorflow weight initialization.
    W = []
    for layer in layers:
        np.random.seed(seed)
        w = tf.Variable(np.random.randn(x_train,layer), dtype=tf.float32)
        W.append(w)
        x_train = layer + 1
    return W

# Bias should be included in the weight matrix in the first row.
def add_bias(X, Y):
    ones = tf.ones((tf.shape(X)[0], 1), dtype=tf.float32)
    X = tf.concat([ones, X], axis=1)
    Y = tf.constant(Y, dtype=tf.float32)
    return X, Y

# activations: list of case-insensitive activations strings corresponding to each layer. The possible activations
# are, "linear", "sigmoid", "relu".
def activation_function(activations, val):
    activation_functions = {
        "sigmoid": lambda x: tf.divide(1.0, 1.0 + tf.exp(-x)),
        "relu": lambda x: tf.maximum(0, x),
    }
    #  linear activation 
    activation_fn = activation_functions.get(activations, lambda x: x)
    return activation_fn(val)

# updating Weights with alpha
def update_weights(W, gradient, alpha):
    for (i, grad) in enumerate(gradient):
        W[i].assign_sub(alpha*grad)
    return W

def train_on_batches(X, y, batch_size=32):
    for i in range(0, X.shape[0], batch_size):
        yield X[i:i+batch_size], y[i:i+batch_size]
    #if X.shape[0] % batch_size != 0:
    #    yield X[-(X.shape[0] % batch_size):], y[-(X.shape[0] % batch_size):]

#loss function for calculating svm, mse and cross_entropy
def compute_loss(loss, labels, predictions):
    predictions = tf.transpose(predictions)
    
    if loss == "svm":
        return svm(labels, predictions)
    elif loss == "mse":
        return mse(labels, predictions)
    elif loss == "cross_entropy":
        return cross_entropy(labels, predictions)
    else:
        raise ValueError("Incorrect loss type: " + loss)

# calculate svm loss
def svm(labels, predictions):
    loss_target = tf.maximum(0.0, 1.0 - (labels * predictions))
    svm_loss = tf.reduce_mean(loss_target)
    return svm_loss

#calculate MSE loss
def mse(labels, predictions):
    return tf.reduce_mean(tf.square(labels - predictions))

#Calculate Cross_Entropy loss
def cross_entropy(labels, predictions):
    cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=predictions)
    return tf.reduce_mean(cross_entropy_loss)

# Each output weights in this assignment is the transpose of the output weights
def actual_network_output(out):
    output = tf.transpose(out)
    ones_row = tf.ones(shape=(tf.shape(output)[0], 1), dtype=tf.float32)
    output = tf.concat([ones_row, output], axis=1)
    return output

# The weights in this assignment are the transpose of the weights
def forward_pass(X, W, activation):
    inputs = X
    for i in range(len(W)):
        #Calculate Output
        o_1 = tf.matmul(tf.transpose(W[i]), tf.transpose(inputs))
        output = activation_function(activation[i], o_1)
        if i + 1 <= len(W):
            inputs = actual_network_output(output)
    return output

# Implementation of Multi_layer_Neural_Network    
def multi_layer_nn_tensorflow(X_train,Y_train,layers,activations,alpha,batch_size,epochs=1,loss="svm",
                        validation_split=[0.8,1.0],seed=2):
    # validation_split: a two-element list specifying the normalized start and end point to
    # extract validation set. Use floor in case of non integers.

    # I have used int to round down in case of non-integers
    start = int(validation_split[0] * X_train.shape[0])
    end = int(validation_split[1] * X_train.shape[0])
    X_val, Y_true = X_train[start:end], Y_train[start:end]
    X_train, Y_train = np.concatenate((X_train[:start], X_train[end:])), np.concatenate((Y_train[:start], Y_train[end:]))

    # Add bias to the training data
    X, Y = add_bias(X_train, Y_train)

    # Add bias to the validation data
    X_val, Y_true = add_bias(X_val, Y_true)

    # Get the number of input features
    x_train = tf.shape(X)[1].numpy()


    # layers: Either a list of integers or a list of numpy weight matrices.
    # If layers is a list of integers then it represents number of nodes in each layer. In this case
    # the weight matrices should be initialized by random numbers.
    # If the layers is given as a list of weight matrices, then the given matrices should be used and NO random
    # initialization is needed.
    
    if all(isinstance(layer, np.ndarray) for layer in layers):
        weights = layers 
        #print("layers=",W)
    else:
        weights = Weight_Matrix(layers, x_train, seed)
        #print("weight values=",W)
        
    err = []
    #number of epochs for training.
    for epoch in range(epochs):  

        # Use minibatch to calculate error and adjust the weights
        # Train on Batch

        #Below are steps for training Multilayer_nn in tensorflow: Forward pass, compute loss, back propagation, update weights
        for (X,Y) in train_on_batches(X, Y, batch_size):    
            with tf.GradientTape() as tape:
                predictions = forward_pass(X, weights, activations)    
                error = compute_loss(loss, Y, predictions)      
                #print("error=",error)
            grads = tape.gradient(error, weights)                 
            weights = update_weights(weights, grads, alpha)             

        # Use steepest descent for adjusting the weights
        Y_pred = forward_pass(X_val, weights, activations) 
        
        # The second element should be a one dimensional list of numbers
        # representing the error after each epoch.     
        error = compute_loss(loss, Y_true, Y_pred)        
        err.append(error.numpy())                         
    
    # The third element should be a two-dimensional numpy array [nof_validation_samples,output_dimensions]
    # representing the actual output of the network when validation set is used as input.
    Out = forward_pass(X_val,weights,activations).numpy()        
    #print("output=",Out)
    
    #The first element of the return list should be a list of weight matrices.
    W = []
    for w in weights:                           
        W.append(w)

    return [W, err, np.transpose(Out)]
   
# I have referred my assignment-1 for helper functions and have implemented as required in tensorflow. Also referred the tensorflow tutorials provided in website for understanding and writing basic code.

