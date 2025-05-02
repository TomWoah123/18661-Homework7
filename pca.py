from scipy.io import loadmat
from matplotlib import pyplot as plt
import numpy as np
import math

def pca_fun(input_data, target_k): 
    """input_data: n x d array containing n d-dimensional data points (array),
    target_d: target dimensionality k < d (int>0),
    returns: d x k matrix containing k eigenvectors (array)
    """
    # TODO: YOUR CODE HERE
    n = input_data.shape[0]
    average_input = np.mean(input_data, axis=0)
    zero_centered_input = input_data - average_input
    covariance_matrix = 1 / n * (zero_centered_input.T @ zero_centered_input)
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    eigenvectors = eigenvectors.transpose()
    eigenpairs = list(zip(eigenvalues, eigenvectors))
    eigenpairs.sort(key=lambda k: k[0], reverse=True)
    top_k_pairs = eigenpairs[:target_k]
    top_k_eigenvectors = [vector for value, vector in top_k_pairs]
    p_matrix = np.array(top_k_eigenvectors).T
    return p_matrix

# Example of plots
top_k=10
cols = 5   # Number of columns
rows = math.ceil(top_k/cols)  # Number of rows
# Create the grid layout for original fgnet
original_faces_grid = np.zeros((50 * rows, 50 * cols))
raw_image = loadmat('face_data.mat')['image'][0]

# Fill the grid with original fgnet
for i in range(top_k):
    row = i // cols  # Determine the row
    col = i % cols   # Determine the column
    original_faces_grid[row * 50:(row + 1) * 50, col * 50:(col + 1) * 50] = raw_image[i+20]

# Plot the grid of original fgnet
plt.figure(figsize=(10, 10))
plt.imshow(original_faces_grid, cmap='gray')
plt.axis('off')
plt.title(f"Original Faces: {rows} x {cols}")
plt.show()

# Representing the image dataset as a matrix
input_data_matrix = np.zeros(shape=(640, 2500))
for i in range(len(raw_image)):
    input_data_matrix[i, :] = raw_image[i].reshape((2500,))

reduced_matrix = pca_fun(input_data_matrix, 200)
print(reduced_matrix.shape)

eigenfaces_grid = np.zeros((50 * rows, 50 * cols))
for i in range(top_k):
    row = i // cols  # Determine the row
    col = i % cols   # Determine the column
    eigenfaces_grid[row * 50:(row + 1) * 50, col * 50:(col + 1) * 50] = \
        reduced_matrix[:, i].reshape((50, 50))

plt.figure(figsize=(10, 10))
plt.imshow(eigenfaces_grid, cmap='gray')
plt.axis('off')
plt.title(f"Eigenfaces: {rows} x {cols}")
plt.show()
