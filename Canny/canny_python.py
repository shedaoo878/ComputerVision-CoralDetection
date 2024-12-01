import cv2
import numpy as np
from skimage import io

def canny_edge_detection(image):
    # blur
    blurred = cv2.GaussianBlur(image, (7,7), 3)
    
    _, binary = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)

    #applying Canny Edge Detection with 50 and 120 as the threshold
    edges = cv2.Canny(binary, 50, 120)

    return edges

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
    
    edges = canny_edge_detection(image_resized)

    edges = cv2.resize(edges, (int(image.shape[0]), int(image.shape[1])))

    # write image to a jpg file
    while True:
        try:
            file_result_name = input("A .jpg file will be added to CannyResults. Give a name for the file: ")
            cv2.imwrite("CannyResults/"+file_result_name, edges)
        except cv2.error as e:
            print("Error writing image:", e)
        else:
            print("Image written successfully!")
            break

    #show Canny Edge Detection
    io.imshow(edges)
    io.show()