import numpy as np 

def generate_rendle_dummy_dataset():
  # Example dummy data from Rendle 2010 
  # Categorical variables (Users, Movies, Last Rated) have been one-hot-encoded 
  x_data = np.matrix([
  #    Users  |     Movies     |    Movie Ratings   | Time | Last Movies Rated
  #   A  B  C | TI  NH  SW  ST | TI   NH   SW   ST  |      | TI  NH  SW  ST
      [1, 0, 0,  1,  0,  0,  0,   0.3, 0.3, 0.3, 0,     13,   0,  0,  0,  0 ],
      [1, 0, 0,  0,  1,  0,  0,   0.3, 0.3, 0.3, 0,     14,   1,  0,  0,  0 ],
      [1, 0, 0,  0,  0,  1,  0,   0.3, 0.3, 0.3, 0,     16,   0,  1,  0,  0 ],
      [0, 1, 0,  0,  0,  1,  0,   0,   0,   0.5, 0.5,   5,    0,  0,  0,  0 ],
      [0, 1, 0,  0,  0,  0,  1,   0,   0,   0.5, 0.5,   8,    0,  0,  1,  0 ],
      [0, 0, 1,  1,  0,  0,  0,   0.5, 0,   0.5, 0,     9,    0,  0,  0,  0 ],
      [0, 0, 1,  0,  0,  1,  0,   0.5, 0,   0.5, 0,     12,   1,  0,  0,  0 ]
  ])
  # ratings
  y_data = np.array([5, 3, 1, 4, 5, 1, 5])

  # Let's add an axis to make tensoflow happy.
  y_data.shape += (1, )

  print("Returning Rendle Dummy Dataset")
  return x_data, y_data





























