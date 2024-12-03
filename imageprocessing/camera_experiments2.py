import numpy as np
import cv2
import random as rng


cap = cv2.VideoCapture(0)
while(True):
  # Capture frame-by-frame
    ret, frame = cap.read()

   # Our operations on the frame come here
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(9,9),0)
    edges = cv2.Canny(blur, 100, 200)
    ret, thresh_img = cv2.threshold(edges, 100, 255, cv2.THRESH_OTSU)

    contours =  cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]

    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    centers = [None]*len(contours)
    radius = [None]*len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.minAreaRect(contours_poly[i])
    for i in range(len(contours)):
        color = (0,0,255)
        cv2.drawContours(frame, contours_poly, i, color)
        cv2.rectangle(frame, (int(boundRect[i][0]), int(boundRect[i][1])), \
        (int(boundRect[i][0]+boundRect[i][2]), int(boundRect[i][1]+boundRect[i][3])), color, 2)
    # for c in contours:
    #     cv2.drawContours(frame, [c], 0, (255,0,0), 3)

     # Display the resulting frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()



# import cv2
# import numpy as np
# import cv2 as cv
# from matplotlib import pyplot as plt

# def main(): 
#     # Open the default webcam  
#     cap = cv2.VideoCapture(0) 
      
#     while True: 
#         # Read a frame from the webcam 
#         ret, frame = cap.read() 
#         if not ret: 
#             print('Image not captured') 
#             break
          
#         # Perform Canny edge detection on the frame 
#         blurred, edges = canny_edge_detection(frame) 
#         contour_detection(edges, blurred)
#         # Display the original frame and the edge-detected frame 
#         #cv2.imshow("Original", frame) 
#         cv2.imshow("Blurred", blurred) 
#         cv2.imshow("Edges", edges) 
          
#         # Exit the loop when 'q' key is pressed 
#         if cv2.waitKey(1) & 0xFF == ord('q'): 
#             break
      
#     # Release the webcam and close the windows 
#     cap.release() 
#     cv2.destroyAllWindows()


# def canny_edge_detection(frame): 
#     # Convert the frame to grayscale for edge detection 
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
      
#     # Apply Gaussian blur to reduce noise and smoothen edges 
#     blurred = cv2.GaussianBlur(src=gray, ksize=(3, 5), sigmaX=0.5) 
      
#     # Perform Canny edge detection 
#     edges = cv2.Canny(blurred, 70, 135) 
      
#     return blurred, edges


# def contour_detection(canny, img):
#     contours, hierarchy = cv2.findContours(canny,
#     cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
#     print("Number of Contours = " ,len(contours))

#     cv2.drawContours(img, contours, -1, (0, 255, 0), 3)

#     cv2.imshow('Contours', img)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()


# if __name__ == "__main__":
#     main()

