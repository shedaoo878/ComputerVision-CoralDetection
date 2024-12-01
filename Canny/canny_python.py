#canny edge detection

import cv2
import numpy as np

def canny_edge_detection(image):
    gray = image
    

    # blur
    blurred = cv2.GaussianBlur(gray, (7,7), 3)
    #binary thresholding
    _, binary = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY)
    cv2.imshow("Preprocessed", binary)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    edges = cv2.Canny(binary, 30,90)
    
    #show image
    cv2.imshow("Edges", edges)
    cv2.imwrite("CannyResults/cannyResult.jpg", edges);
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":  
    image = cv2.imread("../Images/coraltest.pgm", cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Error: Could not load image.pgm")
        exit(1)
    
    edges = canny_edge_detection(image)