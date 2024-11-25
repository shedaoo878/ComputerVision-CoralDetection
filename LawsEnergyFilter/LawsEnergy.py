import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import io

# Define Laws kernels
L5 = np.array([[1, 4, 6, 4, 1]])
E5 = np.array([[-1, -2, 0, 2, 1]])
S5 = np.array([[-1, 0, 2, 0, -1]])

# Load the grayscale image
img = cv2.imread("Images/coral.pgm",  cv2.IMREAD_GRAYSCALE)
io.imshow(img)
io.show()

# Apply Laws filters on the image
L5L5 = cv2.filter2D(img, -1, np.outer(L5, L5))
L5E5 = cv2.filter2D(img, -1, np.outer(L5, E5))
L5S5 = cv2.filter2D(img, -1, np.outer(L5, S5))
E5L5 = cv2.filter2D(img, -1, np.outer(E5, L5))
E5E5 = cv2.filter2D(img, -1, np.outer(E5, E5))
E5S5 = cv2.filter2D(img, -1, np.outer(E5, S5))
S5L5 = cv2.filter2D(img, -1, np.outer(S5, L5))
S5E5 = cv2.filter2D(img, -1, np.outer(S5, E5))
S5S5 = cv2.filter2D(img, -1, np.outer(S5, S5))

# Square the pixel values of each image after filter application to amplify results
L5L5_squared = np.square(L5L5)
L5E5_squared = np.square(L5E5)
L5S5_squared = np.square(L5S5)
E5L5_squared = np.square(E5L5)
E5E5_squared = np.square(E5E5)
E5S5_squared = np.square(E5S5)
S5L5_squared = np.square(S5L5)
S5E5_squared = np.square(S5E5)
S5S5_squared = np.square(S5S5)

# averaging vectors for small neighborhoods 
neighborhood_size = 16
L5L5_avg = cv2.blur(L5L5_squared, (neighborhood_size, neighborhood_size))
L5E5_avg = cv2.blur(L5E5_squared, (neighborhood_size, neighborhood_size))
L5S5_avg = cv2.blur(L5S5_squared, (neighborhood_size, neighborhood_size))
E5L5_avg = cv2.blur(E5L5_squared, (neighborhood_size, neighborhood_size))
E5E5_avg = cv2.blur(E5E5_squared, (neighborhood_size, neighborhood_size))
E5S5_avg = cv2.blur(E5S5_squared, (neighborhood_size, neighborhood_size))
S5L5_avg = cv2.blur(S5L5_squared, (neighborhood_size, neighborhood_size))
S5E5_avg = cv2.blur(S5E5_squared, (neighborhood_size, neighborhood_size))
S5S5_avg = cv2.blur(S5S5_squared, (neighborhood_size, neighborhood_size))

# creating feature vectors
features = np.stack([L5L5_avg.flatten(), L5E5_avg.flatten(), L5S5_avg.flatten(),
                     E5L5_avg.flatten(), E5E5_avg.flatten(), E5S5_avg.flatten(),
                     S5L5_avg.flatten(), S5E5_avg.flatten(), S5S5_avg.flatten()], axis=1)


# Apply K-Means 
kmeans = KMeans(n_clusters=4, random_state=0).fit(features)
labels = kmeans.labels_

# Reshape labels to match the image shape
labels = labels.reshape(img.shape)

# Visualize the segmented image
io.imshow(labels, cmap='rainbow')
io.show()