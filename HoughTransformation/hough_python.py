

import numpy as np
import cv2

def hough_transform(image, min_radius, max_radius):
    if image is None:
        raise ValueError("Could not load image")
    
    gray = image


    _, binary = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    
   
    blurred = cv2.GaussianBlur(binary, (9, 9), 6)
    


    cv2.imshow("Original Image", gray)
    cv2.waitKey(0)
   

    cv2.imshow("Binary and Gaussian Blurred", blurred)
    cv2.waitKey(0)
    
    

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
    image = cv2.imread("Images/coraltest.pgm", cv2.IMREAD_UNCHANGED)
    
    if image is None:
        print("Error: Could not load image.pgm")
        exit(1)
        
    min_radius = 10 
    max_radius = 25
    
    try:
        result = hough_transform(image, min_radius, max_radius)
        
        
        cv2.imshow("Detected Circles", result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error processing image: {str(e)}")