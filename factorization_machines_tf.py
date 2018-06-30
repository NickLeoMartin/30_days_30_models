"""
Factorization Machines for Regression 
"""
import numpy as np
import tensorflow as tf

## TODO: name scope & setting devices

class FactorizationMachines(object):
  """
  Paper: https://www.csie.ntu.edu.tw/~b97053/paper/Rendle2010FM.pdf
  
  Example:
  --------
  from factorization_machines_tf import FactorizationMachines
  from utils import generate_rendle_style_dataset

  x_data, y_data = generate_rendle_style_dataset()
  
  fm = FactorizationMachines(l_factors=10)
  fm.fit(x_data, y_data)
  fm.predict(x_data)

  """
  def __init__(self, l_factors=10, reg_w=0.01, reg_v=0.01, learning_rate=0.01, batch_size=5,
        num_epochs=1000, verbose=True):
    self.l_factors = l_factors
    self.reg_w = reg_w
    self.reg_v = reg_v
    self.num_epochs = num_epochs
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.verbose = verbose
    self.graph = tf.Graph()

  def get_model(self, X_shape):
    """
    Setup Computation Graph
    """
    self.n_dim, self.p_dim = X_shape

    with self.graph.as_default():
      ## Learning rate
      eta = tf.constant(self.learning_rate)

      ## Define placeholders for X and Y
      X = tf.placeholder('float', shape=[None, self.p_dim],name='X')
      Y = tf.placeholder('float', shape=[None, 1])

      ## Define bias, weight matrix, interaction factors and y variables 
      w0 = tf.Variable(tf.zeros([1]))  
      W = tf.Variable(tf.zeros([self.p_dim])) 
      V = tf.Variable(tf.random_normal([self.l_factors, self.p_dim], stddev=0.01))
      Y_pred = tf.Variable(tf.zeros([self.n_dim, 1]),name='Y_pred')

      ## Linear and interaction parameters
      linear_model = tf.add(w0, tf.reduce_sum(tf.multiply(W,X),1,keepdims=True))
      interaction_model = (tf.multiply(0.5,
            tf.reduce_sum(
              tf.subtract(tf.pow(
                tf.matmul(X,tf.transpose(V)),2),
                tf.matmul(tf.pow(X,2), tf.transpose(tf.pow(V,2)))),1, keepdims=True)))

      ## Predictions
      Y_pred = tf.add(linear_model, interaction_model,name="predictions")

      ## Regularization terms
      lambda_w = tf.constant(self.reg_w, name='lambda_w')
      lambda_v = tf.constant(self.reg_v, name='lambda_v')
      l2_norm = (tf.reduce_sum(
          tf.add(
            tf.multiply(lambda_w, tf.pow(W, 2)),
            tf.multiply(lambda_v, tf.pow(V, 2)))
          ))

      ## Error term
      error = tf.reduce_sum(tf.square(tf.subtract(Y, Y_pred)))
      cost = tf.add(error, l2_norm)

      ## Optimizer
      optimizer = tf.train.AdagradOptimizer(eta).minimize(cost)

      ## Initialize variable op
      init = tf.global_variables_initializer()

      ## Return ops and variables
      return error, cost, optimizer, init, W, V, X, Y, Y_pred,

  def fit(self, X_data, Y_data):
    """ 
    Train model
    """
    ## Get ops and variables
    error, cost, optimizer, init,\
    W, V, X, Y, Y_pred = self.get_model(X_data.shape)

    ## Set session
    sess = tf.Session(graph=self.graph)

    ## Train model
    with sess.as_default():
      ## Initialize variables
      sess.run(init)

      ## Train for num_epochs
      for epoch in range(self.num_epochs):
        indices = np.arange(self.batch_size)
        np.random.shuffle(indices)
        x_batch, y_batch = X_data[indices], Y_data[indices]
        batch_feed_dict = {X: x_batch, Y: y_batch}
        sess.run(optimizer, feed_dict=batch_feed_dict)

        if self.verbose:
          ## Retrieve metrics, predictions and weights
          batch_mse, batch_loss = sess.run([error, cost], feed_dict=batch_feed_dict)
          print(f"Epoch: {epoch} MSE : {batch_mse:.3f} Loss : {batch_loss:.3f}")

    ## Set variables
    self.sess = sess

  def predict(self, X_data):
    """ 
    Predict using trained model
    """
    with self.sess.as_default() as sess:
      Y_pred_op = self.graph.get_tensor_by_name("predictions:0")
      X_op = self.graph.get_tensor_by_name("X:0")
      preds = sess.run(Y_pred_op, feed_dict={X_op: X_data})

    if self.verbose:
      print(f"Test set predictions : {preds}")

