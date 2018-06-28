"""
Support Vector Machines for Classification
"""
import numpy as np
import tensorflow as tf

## TODO: Name scope & setting devices
## TODO: Add RBF kernel option for non-linear svm
## TODO: Add multi-class classification with One-versus-Rest classifiers

class SupportVectorMachines(object):
  """
  Binary class SVM with l2-normalization of the weights

  Example:
  --------
  from support_vector_machines_tf import SupportVectorMachines
  from utils import generate_classification_style_dataset

  x_data, y_data = generate_classification_style_dataset("binary")
  
  fm = SupportVectorMachines()
  fm.fit(x_data, y_data)
  fm.predict(x_data)

  """
  def __init__(self, alpha=0.01, learning_rate=0.01, batch_size=5, num_epochs=1000, verbose=True):
    self.alpha = alpha
    self.learning_rate = learning_rate
    self.batch_size = batch_size
    self.num_epochs = num_epochs
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
      X = tf.placeholder('float', shape=[None, self.p_dim],name="X")
      Y = tf.placeholder('float', shape=[None, 1], name="Y")

      ## Define bias, weight matrix, interaction factors and y variables 
      B = tf.Variable(tf.random_normal(shape=[1,1]))
      W = tf.Variable(tf.random_normal(shape=[self.p_dim,1]))
      alpha = tf.constant([self.alpha])
      Y_pred = tf.Variable(tf.zeros([self.p_dim, 1]))
      print("Vars specified")

      ## Model output
      model_output = tf.subtract(tf.matmul(X,W),B)

      ## l2-norm
      l2_norm = tf.reduce_sum(tf.square(W))

      ## Define max-margin loss function
      loss = tf.add(
        tf.reduce_sum(
          tf.maximum(0., tf.subtract(1.,tf.multiply(model_output,Y)))
          ), 
        tf.multiply(alpha,l2_norm)) 

      ## Predictions
      Y_pred = tf.sign(model_output)

      ## Accuracy 
      accuracy = tf.divide(tf.reduce_sum(tf.cast(tf.equal(Y_pred,Y),tf.float32)),self.n_dim)

      ## Optimizer
      optimizer = tf.train.GradientDescentOptimizer(eta).minimize(loss)

      ## Initialize variable op
      init = tf.global_variables_initializer()

      ## Return ops and variables
      return accuracy, loss, optimizer, init, W, X, Y, Y_pred

  def fit(self, X_data, Y_data):
    """ 
    Train model
    """
    ## Get ops and variables
    accuracy, loss, optimizer, init,\
    W, X, Y, Y_pred = self.get_model(X_data.shape)

    ## Set session
    sess = tf.Session(graph=self.graph)

    ## Train model
    with sess.as_default():
      ## Initialize variables
      sess.run(init)

      ## Train for num_epochs
      for epoch in range(self.num_epochs):
        print(f"Epoch: {epoch}")
        indices = np.arange(self.batch_size)
        np.random.shuffle(indices)
        x_batch, y_batch = X_data[indices], Y_data[indices]
        sess.run(optimizer, feed_dict={X: X_data, Y: Y_data})

      ## Retrieve metrics, predictions and weights
      feed_dict = {X: X_data, Y: Y_data}
      accuracy_ = sess.run(accuracy, feed_dict=feed_dict)
      preds_ = sess.run(Y_pred, feed_dict=feed_dict)
      weights_ = sess.run(W, feed_dict=feed_dict)
      loss_ = sess.run(loss, feed_dict=feed_dict)

    if self.verbose:
      print(f"accuracy : {accuracy_}")
      print(f"loss : {loss_}")
      print(f"predictions : {preds_}")
      print(f"weights : {weights_}")

    ## Set variables
    self.X = X
    self.y_pred = Y_pred
    self.sess = sess

  def predict(self, X_data):
    """ 
    Predict using trained model
    """
    with self.sess.as_default() as sess:
      preds = sess.run(self.y_pred, feed_dict={self.X: X_data})

    if self.verbose:
      print(f"Test set predictions : {preds}")

