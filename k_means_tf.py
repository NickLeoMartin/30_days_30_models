"""
K-means Clustering Algorithm
"""
import numpy as np
import tensorflow as tf

## TODO: name scope & setting devices

class KMeans(object):
  """
  K-means Clustering Algorithm
  
  Example:
  --------
  from k_means_tf import KMeans
  from utils import generate_rendle_style_dataset
  
  ## Get dummy data
  x_data, _ = generate_rendle_style_dataset()
  
  ## Fit and predict 
  fm = KMeans(n_clusters=3)
  fm.fit(x_data)
  fm.predict(x_data)

  """
  def __init__(self, n_clusters=3, num_epochs=100, verbose=True):
    self.n_clusters = n_clusters
    self.num_epochs = num_epochs
    self.verbose = verbose
    self.graph = tf.Graph()

  def get_model(self, X_data):
    """
    Setup Computation Graph
    """
    self.n_dim, self.p_dim = X_data.shape

    with self.graph.as_default():

      ## Define placeholders for X and Y
      X = tf.placeholder('float64', shape=[None, self.p_dim],name='X')
      cluster_labels = tf.Variable(tf.zeros([self.n_dim],dtype=tf.int64))

      ## Sample starting points as centroids
      init_points = np.array([X_data[np.random.choice(self.n_dim)] for _ in range(self.n_clusters)])
      centroids = tf.Variable(init_points)

      ## Obtain cluster groupings by minimising distances with centroids
      centroid_matrix = tf.reshape(tf.tile(tf.squeeze(centroids), [self.n_dim,1]), [self.n_dim,self.n_clusters,self.p_dim])
      point_matrix = tf.reshape(tf.tile(X, [1,self.n_clusters]), [self.n_dim,self.n_clusters, self.p_dim])
      distances = tf.reduce_sum(tf.square(point_matrix - centroid_matrix), axis=2)
      cluster_group = tf.argmin(distances, 1, name="cluster_group")

      ## Average clusters to obtain new centroids
      cluster_means = tf.divide(
        tf.unsorted_segment_sum(X, cluster_group, 3),
        tf.unsorted_segment_sum(tf.ones_like(X), cluster_group, 3)
        ) 

      ## Update
      update = tf.group(centroids.assign(tf.expand_dims(cluster_means,1)),cluster_labels.assign(cluster_group))

      ## Initialize variable op
      init = tf.global_variables_initializer()

      ## Return ops and variables
      return init, update, cluster_group, X

  def fit(self, X_data):
    """ 
    Train model
    """
    ## Get ops and variables
    init, update, cluster_group, X = self.get_model(X_data)

    ## Set session
    sess = tf.Session(graph=self.graph)

    ## Train model
    with sess.as_default():
      ## Initialize variables
      sess.run(init)

      ## Train for num_epochs
      for epoch in range(self.num_epochs):
        batch_feed_dict = {X: X_data}
        _, clust_group = sess.run([update, cluster_group], feed_dict=batch_feed_dict)

        if self.verbose:
          print(f"Epoch: {epoch} Labels: {clust_group}")

    ## Set variables
    self.sess = sess

  def predict(self, X_data):
    """ 
    Predict using trained model
    """
    with self.sess.as_default() as sess:
      cluster_group_op = self.graph.get_tensor_by_name("cluster_group:0")
      X_op = self.graph.get_tensor_by_name("X:0")
      pred_groups = sess.run(cluster_group_op, feed_dict={X_op: X_data})

    if self.verbose:
      print(f"Test set cluster groups : {pred_groups}")

    return pred_groups











