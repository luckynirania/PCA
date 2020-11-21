import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio
import copy

if len(sys.argv) < 3:
    print("usuage : python3 program.py <dataset-path> <n_components>")
    sys.exit()

n_components = int(sys.argv[2])

class PCA:
    n_components = 0
    components_ = []
    mean_record = []
    def __init__(self, n_components):
        self.n_components = n_components
    # will give us n_components principal components
    def apply(self, mat):
        # making copy our matrix
        matrix = copy.deepcopy(mat.to_numpy())

        # column mean subtraction
        self.mean_record = matrix.mean(axis=0, keepdims=True)
        X = matrix - matrix.mean(axis=0, keepdims=True)

        print(X.shape)

        C = np.cov(X,rowvar=False)

        print(C.shape)

        u, s, vh = np.linalg.svd(C, hermitian=True)

        print(u)
        print(s)
        print(u.shape, s.shape, vh.shape)

        pca_mat = u[:,:10]
        print('test', pca_mat.shape)
        k_mat = u[:,:self.n_components]

        print(k_mat.shape)

        self.components_ = pca_mat.transpose()
        self.projected = np.dot(np.dot(X, k_mat), k_mat.transpose()) + self.mean_record

image_path = sys.argv[1]

image_files = sorted([os.path.join(image_path, file)
    for file in os.listdir(image_path) if file.endswith('.png')])

# print(image_files)
faces = pd.DataFrame([])

for each in image_files:
    img = imageio.imread(each)
    face = pd.Series(img.flatten(),name=each)
    faces = faces.append(face)

print(faces.iloc[0].values.reshape(64,64))

fig, axes = plt.subplots(3,10,figsize=(9,3), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.iloc[i].values.reshape(64,64),cmap="gray")
fig.suptitle('Original Images')
# plt.show()


#n_components=0.80 means it will return the Eigenvectors that have the 80% of the variation in the dataset
faces_pca = PCA(n_components)
faces_pca.apply(faces)

# print(faces_pca.components_[0].reshape(64,64))

fig, axes = plt.subplots(1,10,figsize=(9,3), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces_pca.components_[i].reshape(64,64),cmap="gray")
fig.suptitle('Visualization of 10 principal components')
# plt.show()

# components = faces_pca.transform(faces)
# projected = faces_pca.inverse_transform(components)
fig, axes = plt.subplots(3,10,figsize=(9,3), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces_pca.projected[i].reshape(64,64),cmap="gray")
fig.suptitle('Reconstruction using ' + str(n_components) + ' componentes')
plt.show()