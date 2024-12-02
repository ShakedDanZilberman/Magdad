import cv2
import numpy as np


cam = cv2.VideoCapture(1)

cv2.namedWindow("test")

img_counter = 4

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1

cam.release()

cv2.destroyAllWindows()

before_change = cv2.imread('opencv_frame_2.png')
cv2.imshow('image', before_change) 

# To close the window 
cv2.waitKey(0) 
cv2.destroyAllWindows() 
after_change = cv2.imread('opencv_frame_3.png')
cv2.imshow('image', after_change) 

# To close the window 
cv2.waitKey(0) 
cv2.destroyAllWindows() 

background = cv2.imread('opencv_frame_4.png')

change = cv2.subtract(after_change, before_change)
displayed = cv2.add(background, change)
cv2.imshow('image', displayed) 

# To close the window 
cv2.waitKey(0) 
cv2.destroyAllWindows() 


# import numpy as np
# import cv2 
# import time
# import os


# cam = cv2.VideoCapture(0)
# if not cam.isOpened():
#     print("Cannot open camera")
#     exit()
# while True:
#     # Capture frame-by-frame
#     ret, frame = cam.read()
 
#     # if frame is read correctly ret is True
#     if not ret:
#         print("Can't receive frame (stream end?). Exiting ...")
#         break
#     # Our operations on the frame come here
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Display the resulting frame
#     cv2.imshow('frame', gray)
#     if cv2.waitKey(1) == ord('q'):
#         break
 
# # When everything done, release the capture
# cam.release()
# cv2.destroyAllWindows()


# cam = cv2.VideoCapture(0)

# frameno = 0
# while(True):
#    ret,frame = cam.read()
#    if ret:
#       # if video is still left continue creating images
#       name = str(frameno) + '.jpg'
#       print ('new frame captured...' + name)

#       cv2.imwrite(name, frame)
#       frameno += 1
#    else:
#       break

# cam.release()
# cv2.destroyAllWindows()