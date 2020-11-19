import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imageio

image_path = "Datasets/Group 6 Prateek Jain and Lokesh Kumar Nirania"

image_files = sorted([os.path.join(image_path, file)
    for file in os.listdir(image_path) if file.endswith('.png')])

# print(image_files)
faces = pd.DataFrame([])

for each in image_files:
    img = imageio.imread(each)
    face = pd.Series(img.flatten(),name=each)
    faces = faces.append(face)

fig, axes = plt.subplots(3,10,figsize=(9,3), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces.iloc[i].values.reshape(64,64),cmap="gray")
plt.show()


from sklearn.decomposition import PCA
#n_components=0.80 means it will return the Eigenvectors that have the 80% of the variation in the dataset
faces_pca = PCA(n_components=10)
faces_pca.fit(faces)

fig, axes = plt.subplots(1,10,figsize=(9,3), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
    ax.imshow(faces_pca.components_[i].reshape(64,64),cmap="gray")
plt.show()

components = faces_pca.transform(faces)
projected = faces_pca.inverse_transform(components)
fig, axes = plt.subplots(3,10,figsize=(9,3), subplot_kw={'xticks':[], 'yticks':[]}, gridspec_kw=dict(hspace=0.01, wspace=0.01))
for i, ax in enumerate(axes.flat):
    ax.imshow(projected[i].reshape(64,64),cmap="gray")
plt.show()