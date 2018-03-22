import cv2
import imutils
import numpy as np
import itertools as it
from numpy.linalg import norm
from matplotlib import pyplot as plt
from utils import *
from keyDetection import detect_black_keys
# class KeyboardDetector():

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")

    sorted_y_indices = np.argsort(pts[:,1])
    rect[0], rect[1] = np.sort(pts[sorted_y_indices[:2],:], axis=0)
    rect[3], rect[2] = np.sort(pts[sorted_y_indices[2:],:], axis=0)

    # return the ordered coordinates
    return rect

def four_point_transform(image, pts):
    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype = "float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def get_warped(image, pts):
    rect = order_points(pts)
    (tl,tr,br,bl) = rect
    # print norm([-320., 0.])
    # a = tl-tr
    # b = np.array([3.,4.])
    # print a, b
    # print "norm(tl-tr)", a.shape, a, b, b.shape, norm(b), norm(a, axis = 0)
    # print "norm(br-bl)", norm(br-bl)
    # print "norm(tl-bl)", norm(tl-bl)
    # print "norm(tr-br)", norm(tr-br)
    if norm(tl-tr, axis = 0) < 20 or norm(br-bl, axis = 0) < 20 or norm(tl-bl, axis = 0) < 20 or norm(tr-br, axis = 0) < 20:
        return None
    width_top = norm(tl-tr, axis = 0)
    width_bottom = norm(bl-br, axis = 0)
    width = max(int(width_top), int(width_bottom))

    height_left = norm(tl-bl, axis = 0)
    height_right = norm(tr-br, axis = 0)
    height = max(int(height_right), int(height_left))

    destination = np.array([[0,0], [width-1,0], [width-1, height-1],[0,height-1]], dtype = np.float32)

    M = cv2.getPerspectiveTransform(rect, destination)
    warped = cv2.warpPerspective(image, M, (width,height))

    return warped


def check_if_candidate_keyboard(image, maxBlackKeys=0):
    '''
    Checks if the given image is a candidate for a keyboard
    Returns:
        Number of black keys found in the top two third of the image
    '''
    H, W, _ = image.shape
    if H < 10:
        return

    H2_3 = int(2./3 * H)
    top_third = image[0:H2_3, :]
    lower_third = image[H2_3:,:]

    top_third_gray = cv2.cvtColor(top_third, cv2.COLOR_BGR2GRAY)
    lower_third_gray = cv2.cvtColor(lower_third, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(top_third_gray,(5,5),0)
    _,top_thresh = cv2.threshold(top_third_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    blur = cv2.GaussianBlur(lower_third_gray,(5,5),0)
    _,lower_thresh = cv2.threshold(lower_third_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)


    if np.mean(top_thresh) < np.mean(lower_thresh):
        return detect_black_keys(top_third)
    return -1


def detect_keyboard(image):
    image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)

    # ------------- x
    # |
    # |
    # |
    # |
    # |
    # y
    ymax, xmax, _ = image.shape
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 4)
    edged = auto_canny(blurred)

    i = 2
    while True:
        houghLines = cv2.HoughLines(edged, 1, np.pi/180*i, 100)
        if len(houghLines) < 30 or i>10:
            break
        i += 1
    lines = []

    for line in houghLines:
        for rho,theta in line:

            if (theta >= 0.0 and theta < np.pi/180*30):
                continue
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a*rho
            y0 = b*rho
            x1 = int(x0 + 1000*(-b))
            y1 = int(y0 + 1000*(a))
            x2 = int(x0 - 1000*(-b))
            y2 = int(y0 - 1000*(a))


            if abs(90 - 180/np.pi*theta) > 5:
                x1 = int(get_x(rho, theta, 0))
                y1 = 0

                x2 = int(get_x(rho, theta, ymax-1))
                y2 = ymax-1
            else:
                x1 = 0
                y1 = int(get_y(rho, theta, x1))

                x2 = xmax
                y2 = int(get_y(rho, theta, x2))

            # cv2.line(image,(x1,y1),(x2,y2),(0,0,255),2)
            # display_image(image)
            l = Line(Point(x1,y1), Point(x2,y2), rho)
            lines.append(l)
    # display_image(image)

    pairs = list(it.combinations(lines,2))

    maxBlackKeys = 0
    candidate_keyboards = []
    keyboard = None
    for l1, l2 in pairs:
        warped = get_warped(image, np.array([(l1.start.x, l1.start.y), (l1.end.x, l1.end.y), (l2.start.x, l2.start.y), (l2.end.x, l2.end.y)]))
        if warped is not None:
            # display_image(warped)
            # candidate_keyboards.append(warped)

            num_black_keys = check_if_candidate_keyboard(warped, maxBlackKeys)
            if num_black_keys >= maxBlackKeys:
                # candidate_keyboards.append(warped)
                maxBlackKeys = num_black_keys
                keyboard = warped
    # for im in candidate_keyboards:
    #     detect_black_keys(im, True)
    #     display_image(im)
    return keyboard






def main():
    # img = cv2.imread('data/arjun1.jpg')
    # img = cv2.imread('keyboard-2.jpg')
    #img = cv2.imread('testimage2.jpg')
    # detectKeyboard(img)
    # readVideo('vid-3.mp4')
    # img = cv2.imread('frame-3.jpg')
    # img = cv2.imread('img.png')

    # img = cv2.imread('img.png')
    # detect_black_keys(img)

    #img = cv2.imread('currFrame.jpg')
    # img = cv2.imread('frame-3.jpg')
    img = cv2.imread('bg.jpg')
    keyboard = detect_keyboard(img)
    if keyboard is not None:
        display_image(keyboard)
    #cv2.imwrite("background"+".jpg", keyboard)
    # readVideo('vid\\sample-jazz-tut-1.mp4')

    # bgd = cv2.imread('bg.jpg')
    # frame = cv2.imread('currFrame.jpg')

    # gray = cv2.cvtColor(bgd, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # bgd_thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # frame_thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)[1]

    # positive_diff = frame_thresh-bgd_thresh
    # positive_diff = (positive_diff > 0) * positive_diff
    # # display_image(positive_diff)
    # cv2.imwrite('positive_diff.jpg',positive_diff)

    # gray = cv2.cvtColor(bgd, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # bgd_thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)[1]

    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    # frame_thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY_INV)[1]

    # negative_diff = frame_thresh-bgd_thresh
    # negative_diff = (negative_diff > 0) * negative_diff
    # # display_image(negative_diff)
    # cv2.imwrite('negative_diff.jpg', negative_diff)

    return

if __name__ == '__main__':
    main()
