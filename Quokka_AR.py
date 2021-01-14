""" Example of using OpenCV API to detect and draw checkerboard pattern"""
import numpy as np
import cv2
import glob

# These two imports are for the signal handler
import signal
import sys

#### Some helper functions #####
def reallyDestroyWindow(windowName) :
    ''' Bug in OpenCV's destroyWindow method, so... '''
    ''' This fix is from http://stackoverflow.com/questions/6116564/ '''
    cv2.destroyWindow(windowName)
    for i in range (1,5):
        cv2.waitKey(1) 

def shutdown():
        ''' Call to shutdown camera and windows '''
        global cap
        cap.release()
        reallyDestroyWindow('img')

def signal_handler(signal, frame):
        ''' Signal handler for handling ctrl-c '''
        shutdown()
        sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
##########

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# YOU SHOULD SET THESE VALUES TO REFLECT THE SETUP
# OF YOUR CHECKERBOARD
WIDTH = 6
HEIGHT = 9

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((WIDTH*HEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:HEIGHT,0:WIDTH].T.reshape(-1,2)

# ROLL CAMERA
cap = cv2.VideoCapture(0)

## Step 0: Load the image you wish to overlay
images = glob.glob('quokka.png')  
currentImage = 0  

replaceImg = cv2.imread(images[currentImage])
rows, cols, ch = replaceImg.shape
# Corners of the image
pts1 = np.float32([[0, 0], [cols, 0], [cols, rows], [0, rows]]) 
# Mask Kernel 
kernel = np.ones((3,3), np.uint8)

while (True):

    #capture a frame
    ret, img = cap.read()

    # Our operations on the frame come here
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (HEIGHT, WIDTH), None)


    # If found, add object points, image points (after refining them)
    if ret == True:
        # Corners of the chess board as were MOOOVING.
        pts2 = np.float32([corners[0, 0], corners[8, 0], corners[(WIDTH*HEIGHT)-1, 0], corners[(WIDTH*HEIGHT)-9, 0]])
        
        ## Step 2: Compute the homography
        # Cheating, big cheater, im a cheat( getting transformation matrix from OPENCV instead of computing it)
        # this is even better than findHomography. :)
        H = cv2.getPerspectiveTransform(pts1, pts2)
        rows, cols, _ = img.shape

        # Transforming source image
        tranny_source = cv2.warpPerspective(replaceImg, H, (cols, rows))
        
        ### Step 3: Compute warped mask image
        ret, thresh = cv2.threshold(cv2.cvtColor(tranny_source, cv2.COLOR_BGR2GRAY), 10, 1, cv2.THRESH_BINARY_INV)
        
        ## Step 4: Compute warped overlay image
        mask = cv2.erode(thresh, kernel)
        
        ## Step 5: Compute final image by combining the warped frame with the captured frame
        for c in range(0, 3):
            img[:, :, c] = tranny_source[:, :, c] * (1 - mask[:, :]) + img[:, :, c] * mask[:, :]

    # SHOW ME SOMETHING PRETTY!
    cv2.imshow('img', img)

    # Wait for the key
    key = cv2.waitKey(1)

    if key == ord('q'):  # Quitting the window
        print("Quit")
        break

# COMPUTER VISION!!!!!!!!!!!!
# release everything
shutdown()