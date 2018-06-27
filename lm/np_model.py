import numpy as np

## TODO: Batch iterator for large datasets

class LogisticRegression(object):
  """
  import numpy as np
  from lm.np_model import LogisticRegression
  from utils import generate_classification_style_dataset

  X, Y = generate_classification_style_dataset()
  
  lr = LogisticRegression()
  lr.fit(X, Y)
  lr.predict(X)

  """
  def __init__(self, n_epochs=10, learning_rate=0.1, learning_rate_decay=0.95, l2_reg=0.05, verbose=True):
    self.n_epochs = n_epochs
    self.learning_rate = learning_rate
    self.learning_rate_decay = learning_rate_decay
    self.l2_reg = l2_reg
    self.verbose = verbose

  def activation_func(self, z):
    """ Sigmoid activation function """
    return 1.0 / (1.0 + np.exp((-1)*z))

  def categorical_cross_entropy(self, X, Y):
    """ Categorical cross-entropy """
    ## Forward pass
    z = self.activation_func(np.dot(X,self.W)+self.B)

    ## Calculate cross entropy
    cross_ent = np.mean(np.sum(Y*np.log(z) + (1-Y)*np.log(1-z),axis=1))
    return cross_ent

  def fit(self, X, Y):
    """
    Train model
    """
    ## Check Y is a numpy array. TODO: Check X is numpy array and reshape
    if isinstance(Y,list):
      Y = np.array(Y)

    ## Obtain matrix dimensions
    self.n_classes = np.unique(Y).size 
    self.n_rows, self.n_cols = X.shape

    ## Randomly initialize weights and bias terms
    self.W = np.random.rand(self.n_rows,self.n_classes)
    self.B = np.random.rand(self.n_classes)

    for epoch in range(self.n_epochs):
      ## Forward pass
      probs = self.activation_func(np.dot(X,self.W)+self.B)
      gradients = Y - probs

      ## Weight and bias update
      self.W += self.learning_rate*np.dot(X.T, gradients) - self.learning_rate*self.l2_reg*self.W
      self.B += self.learning_rate*np.mean(gradients,axis=0)

      ## Update learning rate
      self.learning_rate *= self.learning_rate_decay

      ## Display 
      if self.verbose:
        loss = self.categorical_cross_entropy(X,Y)
        print(f"Epoch: {epoch} Loss: {loss}")

  def predict(self, X):
    """
    Predict using trained model
    """
    ## Forward pass
    probs = self.activation_func(np.dot(X,self.W)+self.B)

    ## Argmax on probabilities
    np.putmask(probs, probs>=0.5,1.0)
    np.putmask(probs, probs<0.5,0.0)
    return probs


































