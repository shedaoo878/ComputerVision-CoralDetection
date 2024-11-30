import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from skimage import io
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# patch size for GLCM calculation
PATCH_SIZE = 36

def glcm_features(image, patch_size):
    image //= 64

    features = np.zeros((image.shape[0], image.shape[1], 32))

    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            patch = image[x:x + patch_size + 1, y:y + patch_size + 1]
            glcm = graycomatrix(patch, distances=[3, 5], angles=[0, np.pi/2, np.pi/4, 3*np.pi/4], levels=4, normed=True)
            contrast = graycoprops(glcm, 'contrast').flatten()
            correlation = graycoprops(glcm, 'correlation').flatten()
            energy = graycoprops(glcm, 'energy').flatten()
            homogeneity = graycoprops(glcm, 'homogeneity').flatten()

            features[x, y] = np.concatenate((contrast, correlation, energy, homogeneity))

    return features.reshape(-1, features.shape[-1])

def principle_component_analysis(features):
    pca = PCA()
    pca.fit(features)
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
    pca_features = pca.fit_transform(features)

    print("features: ", len(features))
    print("pca_features: ", len(pca_features))

    return pca_features

def kmeans(clusters, features, image):
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(features)
    labels = kmeans.labels_

    # Reshape labels to match the image shape
    return labels.reshape(image.shape)

if __name__ == "__main__":  
    # Load the image
    image = cv2.imread("../Images/coral.pgm", cv2.IMREAD_GRAYSCALE)

    #decreasing the resolution of the image
    image = cv2.resize(image, (int(image.shape[1]/4), int(image.shape[0]/4)))

    #showing the original image
    io.imshow(image)
    io.show()

    #segmenting the image
    features = glcm_features(image, patch_size=PATCH_SIZE)
    pca_features = principle_component_analysis(features)
    labels = kmeans(clusters=6, features=pca_features, image=image)

    # Visualize the segmented image
    plt.imshow(labels, cmap='rainbow')
    plt.title("GLCM and KMEANS")
    plt.show()

