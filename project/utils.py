import numpy as np
import cv2
import imutils
import glob
from matplotlib import pyplot as plt

def display_image(image):
    cv2.imshow('', image)
    cv2.waitKey(0)
    return

def get_x(rho, theta, y):
    return (rho - y*np.sin(theta)) / np.cos(theta)

def get_y(rho, theta, x):
    return (rho - x*np.cos(theta)) / np.sin(theta)


def degree(theta):
    return (180 * theta) / np.pi

def auto_canny(image, sigma=0.1):
    v = np.median(image)

    lower = int(max(0, (1.0-sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    return edged

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Line():
    '''
    start: type point
    end: type point
    '''
    def __init__(self, start, end, rho):
        self.start = start
        self.end = end
        self.rho = rho

def get_x(rho, theta, y):
    return (rho - y*np.sin(theta)) / np.cos(theta)

def get_y(rho, theta, x):
    return (rho - x*np.cos(theta)) / np.sin(theta)

def get_blobs(top_third):
    top_third_gray = cv2.cvtColor(top_third, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(top_third_gray,(5,5),0)
    _,top_thresh = cv2.threshold(top_third_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    plt.imshow(top_thresh, 'gray')
    plt.show()
    # params = cv2.SimpleBlobDetector_Params()

    # # Change thresholds
    # params.minThreshold = 0;
    # params.maxThreshold = 50;

    # # Filter by Area.
    # # params.filterByArea = True
    # # params.minArea = 1500

    # # Filter by Circularity
    # params.filterByCircularity = True
    # params.maxCircularity = 0.785

    # # Filter by Convexity
    # params.filterByConvexity = True
    # params.minConvexity = 0.87

    # # Filter by Inertia
    # params.filterByInertia = True
    # params.maxInertiaRatio = 0.2

    # # Create a detector with the parameters
    # detector = cv2.SimpleBlobDetector_create(params)
    detector = cv2.SimpleBlobDetector_create()

    # Detect blobs.
    keypoints = detector.detect(top_third_gray)

    print keypoints
    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(top_third, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
         # Show keypoints
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)
    return


'''
class ShapeDetector:
    def __init__(self):
        pass

    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
        if len(approx) == 3:
            shape = "triangle"

        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)

            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a rectangle
            shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

        # if the shape is a pentagon, it will have 5 vertices
        elif len(approx) == 5:
            shape = "pentagon"

        # otherwise, we assume the shape is a circle
        else:
            shape = "circle"

        # return the name of the shape
        return shape
'''
