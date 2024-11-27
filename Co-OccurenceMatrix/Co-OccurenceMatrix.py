import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage import io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

PATCH_SIZE = 36

# Load the image
image = cv2.imread("Images/coral.pgm", cv2.IMREAD_GRAYSCALE)

#decreasing the resolution of the image
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

pca = PCA()
pca.fit(features_shaped)
explained_variance_ratio = pca.explained_variance_ratio_
cumulative_var_ratio = np.cumsum(explained_variance_ratio)

plt.plot(range(1, len(cumulative_var_ratio) + 1), cumulative_var_ratio)
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Scree Plot')
plt.show()

threshold = 0.95
n_components = np.argmax(cumulative_var_ratio >= threshold) + 1

print("n_components: ", n_components)

pca = PCA(n_components=n_components)
pca_features = pca.fit_transform(features_shaped)

print("features: ", len(features_shaped))
print("pca_features: ", len(pca_features))

# Apply K-Means clustering
kmeans = KMeans(n_clusters=6, random_state=0).fit(pca_features)
labels = kmeans.labels_

# Reshape labels to match the image shape
labels = labels.reshape(image.shape)

# Visualize the segmented image
plt.imshow(labels, cmap='rainbow')
plt.title("GLCM and KMEANS")
plt.show()
        
