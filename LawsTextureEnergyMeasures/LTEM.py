import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io
# from scipy.ndimage import uniform_filter, convolve; USE CV2 INSTEAD
from sklearn.decomposition import PCA

def laws_texture_energy(img):
    # Define Laws kernels
    L5 = np.array([[1, 4, 6, 4, 1]])
    E5 = np.array([[-1, -2, 0, 2, 1]])
    S5 = np.array([[-1, 0, 2, 0, -1]])
    R5 = np.array([[1, -4, 6, -4, 1]])
    W5 = np.array([[-1, 2, 0, -2, 1]])

    # Apply Laws filters on the image
    filteredimg = []
    for kernel1 in [L5, E5, S5, R5, W5]:
        for kernel2 in [L5, E5, S5, R5, W5]:
            filteredimg.append(cv2.filter2D(img, -1, np.outer(kernel1, kernel2)))
            # filteredimg.append(convolve(img, np.outer(kernel1, kernel2), mode='nearest'))

    # IMAGE LOOKS BETTER WITHOUT THIS
    # # Square the pixel values of each image after filter application to amplify results
    # for i in range(len(filteredimg)):
    #     filteredimg[i] = np.square(filteredimg[i])

    # averaging vectors for small neighborhoods 
    neighborhood_size = 16
    for i in range(len(filteredimg)):
        filteredimg[i] = cv2.blur(filteredimg[i], (neighborhood_size, neighborhood_size))
        #filteredimg[i] = uniform_filter(filteredimg[i], size=neighborhood_size, mode='nearest')

    # creating feature vectors
    flattened_filteredimg = np.array([arr.flatten() for arr in filteredimg])
    features = np.stack(flattened_filteredimg, axis=1)

    # DEBUGGING
    # #finding the minimum and maximum pixel values after applying changes
    # print("Max", max(features.flatten()))
    # print("Min", min(features.flatten()))
    
    return features


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

    threshold = 0.99
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
    # Load the grayscale image
    img = cv2.imread("../Images/coral.pgm",  cv2.IMREAD_GRAYSCALE)


    #showing the image
    io.imshow(img)
    io.show()

    #using 64 bit float representation
    img = np.array(img, dtype=np.float64)

    # DEBUGGING
    # # Find the minimum and maximum values and their locations.
    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)
    # print("Minimum pixel value:", min_val)
    # print("Minimum pixel location:", min_loc)
    # print("Maximum pixel location:", max_loc)
    # print("Maximum pixel value:", max_val)

    #segmenting the image
    features = laws_texture_energy(img)
    pca_features = principle_component_analysis(features)
    labels = kmeans(clusters=3, features=pca_features, image=img)



    # Visualize the segmented image
    plt.imshow(labels, cmap='rainbow')
    plt.title("LAWS ENERGY FILTERS and KMEANS")
    plt.show()