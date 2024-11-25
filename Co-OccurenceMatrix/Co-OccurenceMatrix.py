import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import io
from sklearn.cluster import KMeans

PATCH_SIZE = 10

# Load the image
image = cv2.imread("Images/coral.pgm", cv2.IMREAD_GRAYSCALE)
image = cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4)))

io.imshow(image)
io.show()

image //= 64

features = np.zeros((image.shape[0], image.shape[1], 32))

for x in range(image.shape[0]):
    for y in range(image.shape[1]):
        patch = image[x:x + PATCH_SIZE + 1, y:y + PATCH_SIZE + 1]
        glcm = graycomatrix(patch, distances=[3, 5], angles=[0, np.pi/2, np.pi/4, 3*np.pi/4], levels=4, normed=True)
        contrast = graycoprops(glcm, 'contrast').flatten()
        correlation = graycoprops(glcm, 'correlation').flatten()
        energy = graycoprops(glcm, 'energy').flatten()
        homogeneity = graycoprops(glcm, 'homogeneity').flatten()

        features[x, y] = np.concatenate((contrast, correlation, energy, homogeneity))

features_shaped = features.reshape(-1, features.shape[-1])

# Apply K-Means clustering with 3 clusters
kmeans = KMeans(n_clusters=5, random_state=0).fit(features_shaped)
labels = kmeans.labels_

# Reshape labels to match the image shape
labels = labels.reshape(image.shape)

# Visualize the segmented image
io.imshow(labels, cmap='rainbow')
io.show()
        
