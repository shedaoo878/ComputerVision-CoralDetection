import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io
# from scipy.ndimage import uniform_filter, convolve; USE CV2 INSTEAD
from sklearn.decomposition import PCA
from matplotlib.colors import Normalize


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
            img = cv2.imread("../Images/"+file_name, cv2.IMREAD_GRAYSCALE)
        except cv2.error as e:
            print("Error reading image:", e)
        else:
            if (img is not None):
                print("Image read successfully!")
                break
            else:
                print("Enter Correct Name")

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

    # segmenting the image
    features = laws_texture_energy(img)
    while True:
        try:
            num_clusters = int(input("Input number of clusters for Kmeans clustering: "))
        except ValueError:
            print("Provide an Integer Value")
        else:
            break
    labels = kmeans(clusters=num_clusters, features=features, image=img)

    # mapping labels to colors
    cmap = plt.get_cmap('viridis', len(np.unique(labels)))
    colored_labels = cmap(labels)
    # convert to RGB
    colored_labels = (colored_labels[:, :, :3] * 255).astype(np.uint8)

    # write image to a jpg file
    while True:
        try:
            file_result_name = input("A .jpg file will be added to LTEMResults. Give a name for the file: ")
            cv2.imwrite("LTEMResults/"+file_result_name, colored_labels)
        except cv2.error as e:
            print("Error writing image:", e)
        else:
            print("Image written successfully!")
            break

    # Visualize the LAWS ENERGY FILTERS and KMEANS image
    io.imshow(colored_labels)
    io.show()