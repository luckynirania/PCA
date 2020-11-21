# importing dependencies required in the code

import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
import copy

# Usage error, in case user passes wrong arguments 
if len(sys.argv) < 3:
    print("usuage : python3 program.py <dataset-path> <n_components>")
    sys.exit()

n_components = int(sys.argv[2])


# Class to perform PCA
class PCA:
    # Default initialization of variables
    n_components = 0 # Stores number of components to be used for PCA
    components_ = [] # Stores first 10 principal components for visualizing the features they represent
    mean_record = [] # Stores the mean of each column in data
    def __init__(self, n_components):
        self.n_components = n_components # Initiallizing with input from user
    def apply(self, mat):
        # making copy of input data
        matrix = copy.deepcopy(mat.to_numpy())

        # centering the data by subtracting column mean
        self.mean_record = matrix.mean(axis=0, keepdims=True)
        X = matrix - matrix.mean(axis=0, keepdims=True)

        print("Shape of Input matrix " + str(X.shape))

        C = np.cov(X,rowvar=False) # Calculating Covariance of input matrix

        print("Shape of Covariance matrix " + str(C.shape))

        u, s, vh = np.linalg.svd(C, hermitian=True) # Performing SVD on Covariance matrix using python library
        # u stores principal components column-wise in descending order of eigen values

        # storing 10 principal components to visualize the features they represent
        pca_mat = u[:,:10] 
        self.components_ = pca_mat.transpose()

        # Performing compression
        k_mat = u[:,:self.n_components] # Storing first n_components
        compressed = np.dot(X, k_mat) # Calculating and storing compressed data

        # Reconstructing the images
        self.reconstructed = np.dot(compressed, k_mat.transpose()) # Recalculating data in uncompressed form
        self.reconstructed += self.mean_record # Adding mean of columns to the data to bring it in original form

# Storing image path
image_path = sys.argv[1]

# Storing the names of all the images in the dataset directory whose path is given by user
image_files = sorted([os.path.join(image_path, file)
    for file in os.listdir(image_path) if file.endswith('.png')])

# Creating a dataframe to store data
faces = pd.DataFrame([])

# Storing the input data
for each in image_files:
    img = imageio.imread(each) # Reading the image
    face = pd.Series(img.flatten(),name=each) # Flattening the image
    faces = faces.append(face) # Appending it to the matrix storing data

# Displaying Original Images
fig, axes = plt.subplots(3,10,figsize=(9,3), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.iloc[i].values.reshape(64,64),cmap="gray")
fig.suptitle('Original Images')


# Performing PCA by calling functions of PCA class
faces_pca = PCA(n_components) # Initializing the object of PCA class
faces_pca.apply(faces) # Calling the function which performs PCA

# Displaying features represented by first 10 principal components
fig, axes = plt.subplots(1,10,figsize=(9,3), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces_pca.components_[i].reshape(64,64),cmap="gray")
fig.suptitle('Visualization of 10 principal components')

# Displaying reconstructed images
fig, axes = plt.subplots(3,10,figsize=(9,3), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces_pca.reconstructed[i].reshape(64,64),cmap="gray")
fig.suptitle('Reconstruction using ' + str(n_components) + ' componentes')
plt.show()