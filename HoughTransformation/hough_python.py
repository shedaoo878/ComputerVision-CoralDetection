

import numpy as np
import cv2
from skimage import io

def hough_transform(image, min_radius, max_radius):

    _, binary = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY_INV)

    blurred = cv2.GaussianBlur(binary, (9, 9), 6)

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=30,    
        param1=40,   #higher val detects stronger circles, lower val detects weaker circles
        param2=20,   #same as above
        minRadius=min_radius,
        maxRadius=max_radius
    )
    
    output = image.copy()
        
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(output, (i[0], i[1]), i[2], (255, 255, 255), 3)
            cv2.circle(output, (i[0], i[1]), 2, (128, 128, 128), 4)
        print(f"Found {len(circles[0])} circles")
    else:
        print("No circles detected")
    
    return output

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

    #showing original image
    io.imshow(image_resized)
    io.show()
        
    #setting minimum and maximum radius
    min_radius = 10 
    max_radius = 25
    
    result = hough_transform(image_resized, min_radius, max_radius)
    
    result = cv2.resize(result, (int(image.shape[0]), int(image.shape[1])))

    # write image to a jpg file
    while True:
        try:
            file_result_name = input("A .jpg file will be added to HoughResults. Give a name for the file: ")
            cv2.imwrite("HoughResults/"+file_result_name, result)
        except cv2.error as e:
            print("Error writing image:", e)
        else:
            print("Image written successfully!")
            break

    #show image
    io.imshow(result)
    io.show()