import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

def apply_laws_filter(img, kernel):
    """Applies a Laws' filter to the input image.

    Args:
        img: The input grayscale image.
        kernel: The Laws' filter kernel.

    Returns:
        The filtered image.
    """

    filtered_img = cv2.filter2D(img, -1, kernel)
    return filtered_img

# Define Laws' kernels
L5 = np.array([[1, 4, 6, 4, 1]])
E5 = np.array([[-1, -2, 0, 2, 1]])
S5 = np.array([[-1, 0, 2, 0, -1]])

# Load the grayscale image
img = cv2.imread("coral.pgm", cv2.IMREAD_GRAYSCALE)

# Apply Laws' filters
L5L5 = apply_laws_filter(img, np.outer(L5, L5))
L5E5 = apply_laws_filter(img, np.outer(L5, E5))
L5S5 = apply_laws_filter(img, np.outer(L5, S5))
E5L5 = apply_laws_filter(img, np.outer(E5, L5))
E5E5 = apply_laws_filter(img, np.outer(E5, E5))
E5S5 = apply_laws_filter(img, np.outer(E5, S5))
S5L5 = apply_laws_filter(img, np.outer(S5, L5))
S5E5 = apply_laws_filter(img, np.outer(S5, E5))
S5S5 = apply_laws_filter(img, np.outer(S5, S5))

# Square the pixel values of each filtered image
L5L5_squared = np.square(L5L5)
L5E5_squared = np.square(L5E5)
L5S5_squared = np.square(L5S5)
E5L5_squared = np.square(E5L5)
E5E5_squared = np.square(E5E5)
E5S5_squared = np.square(E5S5)
S5L5_squared = np.square(S5L5)
S5E5_squared = np.square(S5E5)
S5S5_squared = np.square(S5S5)

# Define neighborhood size
neighborhood_size = 16

# Apply neighborhood averaging
L5L5_avg = cv2.blur(L5L5_squared, (neighborhood_size, neighborhood_size))
L5E5_avg = cv2.blur(L5E5_squared, (neighborhood_size, neighborhood_size))
L5S5_avg = cv2.blur(L5S5_squared, (neighborhood_size, neighborhood_size))
E5L5_avg = cv2.blur(E5L5_squared, (neighborhood_size, neighborhood_size))
E5E5_avg = cv2.blur(E5E5_squared, (neighborhood_size, neighborhood_size))
E5S5_avg = cv2.blur(E5S5_squared, (neighborhood_size, neighborhood_size))
S5L5_avg = cv2.blur(S5L5_squared, (neighborhood_size, neighborhood_size))
S5E5_avg = cv2.blur(S5E5_squared, (neighborhood_size, neighborhood_size))
S5S5_avg = cv2.blur(S5S5_squared, (neighborhood_size, neighborhood_size))

# Combine filtered images into a single feature vector
features = np.stack([L5L5_avg.flatten(), L5E5_avg.flatten(), L5S5_avg.flatten(),
                     E5L5_avg.flatten(), E5E5_avg.flatten(), E5S5_avg.flatten(),
                     S5L5_avg.flatten(), S5E5_avg.flatten(), S5S5_avg.flatten()], axis=1)

# Apply K-Means clustering with 3 clusters
kmeans = KMeans(n_clusters=4, random_state=0).fit(features)
labels = kmeans.labels_

# Reshape labels to match the image shape
labels = labels.reshape(img.shape)

# Visualize the segmented image
plt.imshow(labels, cmap='rainbow')
plt.title('Segmented Image')
plt.show()