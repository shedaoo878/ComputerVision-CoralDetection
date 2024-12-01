import cv2
import numpy as np
from skimage import io

def sobel_edge_detection(image):
    # blur
    blurred = cv2.GaussianBlur(image, (5,5), 3)

    _, image = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)

    # sobel operators
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    # graident magnitude
    magnitude = cv2.magnitude(grad_x, grad_y)

    # normalize magnitude
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return magnitude

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
    image_resized = cv2.resize(image, (int(image.shape[0]/6), int(image.shape[1]/6)))

    #show original image
    io.imshow(image_resized)
    io.show()
    
    sobel = sobel_edge_detection(image_resized)

    sobel = cv2.resize(sobel, (int(image.shape[0]), int(image.shape[1])))

    # write image to a jpg file
    while True:
        try:
            file_result_name = input("A .jpg file will be added to SobelResults. Give a name for the file: ")
            cv2.imwrite("SobelResults/"+file_result_name, sobel)
        except cv2.error as e:
            print("Error writing image:", e)
        else:
            print("Image written successfully!")
            break

    #show Sobel Edge Detection
    io.imshow(sobel)
    io.show()
