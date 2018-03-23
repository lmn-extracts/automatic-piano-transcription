import numpy as np
import cv2
import imutils
import glob
from matplotlib import pyplot as plt


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


def detect_black_keys(img, show=False):
    '''
    image: Color image of the top two third of candidate image

    Returns:
        Maximum number of black keys found
    '''
    image = img[10:,:].copy()


    if show:
        ind = 0
        cv2.imwrite(str(ind) + '.jpg', image)
        ind+=1

    # convert the resized image to grayscale, blur it slightly,
    # and threshold it
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)[1]

    # find contours in the thresholded image and initialize the
    # shape detector
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    sd = ShapeDetector()

    num_black_keys = 0
    black_key_pts = []
    # loop over the contours
    for c in cnts:
        print ("c", c)
        # compute the center of the contour, then detect the name of the
        # shape using only the contour
        M = cv2.moments(c)
        if M["m00"] == 0:
            continue
        cX = int((M["m10"] / M["m00"]))
        cY = int((M["m01"] / M["m00"]))
        shape = sd.detect(c)
        # if shape == "rectangle":
        num_black_keys += 1
        # multiply the contour (x, y)-coordinates by the resize ratio,
        # then draw the contours and the name of the shape on the image
        c = c.astype("float")
        c = c.astype("int")
        # cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
        cv2.fillPoly(image, [c], (0, 255, 0))

        x,y,w,h = cv2.boundingRect(c)

        pts = np.array([[x,y],[x+w,y+h]])

        black_key_corners.append((x,y, w,h))
        # cv2.rectangle(image, (x,y), (x+w-1, y+h-1), (0,255,0), thickness=-1)


        if show:
            cv2.imwrite(str(ind) + '.jpg', image)
            ind+=1
    return num_black_keys, black_key_properties

def assign_while_keys(num_black_keys, black_key_properties):
    #black_keys_pattern = ['D#', 'F#', 'G#', 'A#', 'C#']
    if num_black_keys < 5:
        return
    num_pattern = int(num_black_keys/5)
    black_key_mid_pts = []
    for i in range(6):
        black_key_property = black_key_properties[i]
        black_key_mid_pt = (black_key_property[0]+ black_key_property[2])/2.
        black_key_mid_pts.append(black_key_mid_pt)

    diffs = []
    for i in range(5):
        diff = black_key_mid_pts[i+1] - black_key_mid_pts[i]
        diffs.append((diff, i))

    diffs.sort(lambda x: x[0])
    big_diffs_idx1, big_diffs_idx2 = diffs[-1][1], diffs[-2][1]
    sorted_diffs_idx = sorted[big_diffs_idx1, big_diffs_idx2]
    if abs(big_diffs_idx1 - big_diffs_idx2) == 2:
        if sorted_diffs_idx[0] == 0:
            patten = ['A#', 'C#', 'D#', 'F#', 'G#']
        elif sorted_diffs_idx[0] == 1:
            patten = ['G#', 'A#', 'C#', 'D#', 'F#']
        elif sorted_diffs_idx[0] == 2:
            patten = ['F#', 'G#', 'A#', 'C#', 'D#']
    elif abs(big_diffs_idx1 - big_diffs_idx2) == 3:
        if sorted_diffs_idx[0] == 0:
            pattern = ['D#', 'F#', 'G#', 'A#', 'C#']
        elif sorted_diffs_idx[0] == 1:
            pattern = ['C#', 'D#', 'F#', 'G#', 'A#']
    print(pattern)
    return pattern

def find_white_keys():
    return
