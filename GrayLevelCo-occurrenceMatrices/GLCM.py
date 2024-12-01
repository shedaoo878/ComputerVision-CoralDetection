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

    print("features: ", len(features[0]))
    print("pca_features: ", len(pca_features[0]))

    return pca_features

def kmeans(clusters, features, image):
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=clusters, random_state=0).fit(features)
    labels = kmeans.labels_

    # Reshape labels to match the image shape
    return labels.reshape(image.shape)

if __name__ == "__main__":  

    # Load the image
    while True:
        try:
            file_name = input("Give the file name of an image in the Images folder: ")
            image = cv2.imread("../Images/"+file_name, cv2.IMREAD_GRAYSCALE)
        except cv2.error as e:
            print("Error reading image:", e)
        else:
            if (image is not None):
                print("Image read successfully!")
                break
            else:
                print("Enter Correct Name")
    #decreasing the resolution of the image
    image = cv2.resize(image, (int(image.shape[0]/4), int(image.shape[1]/4)))

    #showing the original image
    io.imshow(image)
    io.show()

    #segmenting the image
    features = glcm_features(image, patch_size=PATCH_SIZE)
    pca_features = principle_component_analysis(features)
    while True:
        try:
            num_clusters = int(input("Input number of clusters for Kmeans clustering: "))
        except ValueError:
            print("Provide an Integer Value")
        else:
            break
    labels = kmeans(clusters=num_clusters, features=pca_features, image=image)

    plt.scatter(pca_features[:, 0], pca_features[:, 1], c=labels, s=.01)
    plt.xlabel('PC0')
    plt.ylabel('PC1')
    plt.title('PCA Clusters')
    while True:
        try:
            file_plot_name = input("A .jpg file will be added to GLCMResults a scatter plot of the PCA Features. Give a name for the file: ")
            plt.savefig('GLCMResults/'+file_plot_name)
        except cv2.error as e:
            print("Error writing image:", e)
        else:
            print("Scatter plot written successfully!")
            break
    plt.show()

    # mapping labels to colors
    cmap = plt.get_cmap('viridis', len(np.unique(labels)))
    colored_labels = cmap(labels)
    # convert to RGB
    colored_labels = (colored_labels[:, :, :3] * 255).astype(np.uint8)

    #write image to a jpg file
    while True:
        try:
            file_result_name = input("A .jpg file will be added to GLCMResults. Give a name for the file: ")
            cv2.imwrite("GLCMResults/"+file_result_name, colored_labels)
        except cv2.error as e:
            print("Error writing image:", e)
        else:
            print("Image written successfully!")
            break

    # Visualize the GLCM and KMEANS image
    io.imshow(colored_labels)
    io.show()