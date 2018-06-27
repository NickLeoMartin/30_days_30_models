import numpy as np 

def generate_rendle_style_dataset():
	"""
	Dummy data from Rendle (2010) 
	"""
	## Categorical variables (Users, Movies, Last Rated) have been one-hot-encoded 
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
	## Ratings
	y_data = np.array([5, 3, 1, 4, 5, 1, 5])

	## Add an axis for tensorflow
	y_data.shape += (1, )

	print("Returning Rendle style dataset")
	return x_data, y_data




def generate_classification_style_dataset():
	"""
	Dummy data to test models
	"""
	x_data = np.array([
		[1,1,1,0,0,0],
		[1,0,1,0,0,0],
		[1,1,1,0,0,0],
		[0,0,1,1,1,0],
		[0,0,1,1,0,0],
		[0,0,1,1,1,0]])
	# y_data = np.array([
	# 	[1, 0],
	# 	[1, 0],
	# 	[1, 0],
	# 	[0, 1],
	# 	[0, 1],
	# 	[0, 1]])

	y_data = np.array([
		[1, 0, 0],
		[1, 0, 0],
		[0, 0, 1],
		[0, 0, 1],
		[0, 1, 0],
		[0, 1, 0]])
	print("Returning classification style dataset")
	return x_data, y_data
























