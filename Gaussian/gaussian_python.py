import cv2
import numpy as np
from skimage import io

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
    
    io.imshow(image)
    io.show()
    
    blurred_image = cv2.GaussianBlur(image, (5, 5), 3)

    # write image to a jpg file
    while True:
        try:
            file_result_name = input("A .jpg file will be added to GaussianResults. Give a name for the file: ")
            cv2.imwrite("GaussianResults/"+file_result_name, blurred_image)
        except cv2.error as e:
            print("Error writing image:", e)
        else:
            print("Image written successfully!")
            break

    io.imshow(blurred_image)
    io.show()