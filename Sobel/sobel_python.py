#sobel edge detection

import cv2
import numpy as np


def sobel_edge_detection(image):
    gray = image

    # blur
    blurred = cv2.GaussianBlur(gray, (5,5), 3)
    #binary thresholding
    _, binary = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
    cv2.imshow("Preprocessed", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    
    # sobel operators
    grad_x = cv2.Sobel(binary, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(binary, cv2.CV_64F, 0, 1, ksize=3)


    # graident magnitude and direction
    magnitude = cv2.magnitude(grad_x, grad_y)

    # normalize magnitude
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    
    return magnitude

if __name__ == "__main__":
    image = cv2.imread("../Images/coraltest.pgm", cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Error: Could not load image.pgm")
        exit(1)
    
    
    magnitude, direction = sobel_edge_detection(image)


    cv2.imshow("Magnitude", magnitude)
    cv2.imshow("Direction", direction)
    cv2.imwrite("SobelResults/sobelResult.jpg", magnitude);
    cv2.waitKey(0)
    cv2.destroyAllWindows()
