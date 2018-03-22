import numpy as np
import cv2
import itertools as it
import imutils
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


def get_Y_indices(line1,line2,img):
	for rho,theta in line1:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))

		if (x2-x1) == 0:
			x1_plot_1 = x1
			y1_plot_1 = 0
			x1_plot_2 = x1_plot_1
			y1_plot_2 = int(img.shape[0])
		else:
			m = (y2-y1) / (x2-x1)
			b = y0 - m * x0
			x1_plot_1 = 0
			y1_plot_1 = int(b)
			x1_plot_2 = int(img.shape[1])
			y1_plot_2 = int(m * x1_plot_2 + b)

	for rho,theta in line2:
		a = np.cos(theta)
		b = np.sin(theta)
		x0 = a*rho
		y0 = b*rho
		x1 = int(x0 + 1000*(-b))
		y1 = int(y0 + 1000*(a))
		x2 = int(x0 - 1000*(-b))
		y2 = int(y0 - 1000*(a))

		if (x2-x1) == 0:
			x2_plot_1 = x1
			y2_plot_1 = 0
			x2_plot_2 = x2_plot_1
			y2_plot_2 = int(img.shape[0])
		else:
			m = (y2-y1) / (x2-x1)
			b = y0 - m * x0
			x2_plot_1 = 0
			y2_plot_1 = int(b)
			x2_plot_2 = int(img.shape[1])
			y2_plot_2 = int(m * x2_plot_2 + b)
	return y1_plot_2, y2_plot_2

def detectKeyboard(img):
	# Resize to 160x120 to make processing faster
	resized = cv2.resize(img, (160,120), cv2.INTER_AREA)
	ratio = img.shape[0] / float(resized.shape[0])

	# Binarize image
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
	sobely = cv2.Scharr(thresh,cv2.CV_64F,0,1)

	# Identify horizontal lines and filter out lines that are off by more than 5 degrees
	lines = cv2.HoughLines(sobely.astype(np.uint8), 1, np.pi/2, 100)
	lines = [l for l in lines if abs(l[0][1]-np.pi/2) < np.pi/180.0*5]

	# Compare brightness of lower one-third and upper one-third to determine the line-pair that crops the keyboard
	pairs = list(it.combinations(lines,2))
	keyboardLines = []
	maxDiff = 0

	# Cache the y-indices of the required horizontal lines to facilitate cropping
	bottomX_cached = 0
	topX_cached = 0

	for (line1,line2) in pairs:
		y1,y2 = get_Y_indices(line1,line2,img)

		topX = min(y1,y2)
		bottomX = max(y1,y2)
		twoThird = int(2./3 * (bottomX-topX) + topX)
		top = gray[topX:twoThird+1,:]
		bottom = gray[twoThird:bottomX+1,:]

		diff = np.mean(bottom) - np.mean(top)
		if  diff >= maxDiff and (bottomX-topX) > (bottomX_cached-topX_cached):
			maxDiff = diff
			keyboardLines = []
			topX_cached = topX
			bottomX_cached = bottomX
			keyboardLines.append(line1)
			keyboardLines.append(line2)

	for line in keyboardLines:
		for rho,theta in line:
		    a = np.cos(theta)
		    b = np.sin(theta)
		    x0 = a*rho
		    y0 = b*rho
		    x1 = int(x0 + 1000*(-b))
		    y1 = int(y0 + 1000*(a))
		    x2 = int(x0 - 1000*(-b))
		    y2 = int(y0 - 1000*(a))

		    if (x2-x1) == 0:
		    	x_plot_1 = x1
		    	y_plot_1 = 0
		    	x_plot_2 = x1
		    	y_plot_2 = int(img.shape[0])
		    else:
		    	m = (y2-y1) / (x2-x1)
		    	b = y0 - m * x0
		    	x_plot_1 = 0
		    	y_plot_1 = int(b)
		    	x_plot_2 = int(img.shape[1])
		    	y_plot_2 = int(m * x_plot_2 + b)

		    #cv2.line(img,(int(x_plot_1*ratio),int(y_plot_1*ratio)), (int(x_plot_2*ratio),int(y_plot_2*ratio)),(0,255,0),2)
		    #cv2.line(resized,(x_plot_1,y_plot_1),(x_plot_2,y_plot_2),(0,255,0),1)
	#cv2.imshow('resized',resized)
	# cv2.imwrite('result.jpg', img[int(topX_cached*ratio):int(bottomX_cached*ratio),:,:])
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	return

def detectShape(c):
	shape = "unidentified"
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04 * peri, True)
	if len(approx) == 4:
		(x,y,w,h) = cv2.boundingRect(approx)
		ar = w / float(h)

		shape = "square" if ar>=0.95 and ar<=1.05 else "rectangle"
	return shape

def detectKeys(img, show=False):
	# Resize to 160x120 to make processing faster
	# resized = cv2.resize(img, (160,120), cv2.INTER_AREA)
	# ratio = img.shape[0] / float(resized.shape[0])
	image = img[10:,:].copy()
	#
	# if show:
	# 	ind = 0
	# 	cv2.imwrite(str(ind) + '.jpg', image)
	# 	ind+=1
	#
	# convert the resized image to grayscale, blur it slightly,
	# and threshold it
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# cv2.imshow('gray',gray)
	# cv2.waitKey(0)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	# cv2.imshow('blurred',blurred)
	# cv2.waitKey(0)
	# edged = cv2.Canny(blurred, 30, 200)
	# cv2.imshow('edged',edged)
	# cv2.waitKey(0)
	thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)[1]
	# cv2.imshow('thresh',thresh)
	# cv2.waitKey(0)
	# find contours in the thresholded image and initialize the
	# shape detector
	sd = ShapeDetector()
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	external_black_key_widths = []
	for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
		M = cv2.moments(c)
		if M["m00"] == 0:
			continue
		cX = int((M["m10"] / M["m00"]))
		cY = int((M["m01"] / M["m00"]))
		shape = sd.detect(c)
		#if shape == "rectangle":
			# multiply the contour (x, y)-coordinates by the resize ratio,
			# then draw the contours and the name of the shape on the image
		c = c.astype("int")
		x,y,w,h = cv2.boundingRect(c)
		external_black_key_widths.append(w)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	all_list_black_key_widths = []
	# loop over the contours
	for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
		M = cv2.moments(c)
		if M["m00"] == 0:
			continue
		cX = int((M["m10"] / M["m00"]))
		cY = int((M["m01"] / M["m00"]))
		shape = sd.detect(c)
		#if shape == "rectangle":
			# multiply the contour (x, y)-coordinates by the resize ratio,
			# then draw the contours and the name of the shape on the image
		c = c.astype("int")
		# cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
		# cv2.fillPoly(image, [c], (0, 255, 0))
		x,y,w,h = cv2.boundingRect(c)
		all_list_black_key_widths.append(w)

	print "external_black_key_widths",external_black_key_widths
	print "all_list_black_key_widths", all_list_black_key_widths
	best_black_key_width = np.bincount(external_black_key_widths+all_list_black_key_widths).argmax()

	black_key_properties = []
	for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
		M = cv2.moments(c)
		if M["m00"] == 0:
			continue
		cX = int((M["m10"] / M["m00"]))
		cY = int((M["m01"] / M["m00"]))
		shape = sd.detect(c)
		#if shape == "rectangle":
			# multiply the contour (x, y)-coordinates by the resize ratio,
			# then draw the contours and the name of the shape on the image
		c = c.astype("int")
		# cv2.drawContours(img, [c], -1, (255,0,0), 3)
		# cv2.imshow('res',img)
		# cv2.waitKey(0)
		#pts = np.array([[x,y],[x+w,y+h]])
		x,y,w,h = cv2.boundingRect(c)
		if abs(w-best_black_key_width) <= 1:
			black_key_properties.append((x,y, w,h))
		# cv2.rectangle(image, (x,y), (x+w-1, y+h-1), (0,255,0), thickness=-1)


	#default black keys is detected from right to left, we reverse to starting from left before output it
	black_key_properties.sort(key = lambda x: x[0])
	if show:
		cv2.imwrite(str(ind) + '.jpg', image)
		ind+=1
	return black_key_properties



	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# blurred = cv2.GaussianBlur(gray, (1,1), 0)
	# thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
	# kernel = np.ones((5,5), np.uint8)
	# thresh = cv2.erode(thresh,kernel,iterations=2)
	#
	# cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	# print len(cnts[0])
	# for c  in cnts[1]:
	# 	shape = detectShape(c)
	# 	print shape
	# 	c = c.astype("float")
	# 	c = c.astype(int)
	# 	if shape == 'rectangle':
	# 		cv2.drawContours(img, [c], -1, (255,0,0), 3)
	# 		cv2.imshow('res',img)
	# 		cv2.waitKey(0)
	#
	#
	# cv2.imshow('res',thresh)
	# cv2.imshow('rese',img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	# return

def assign_white_keys(black_key_properties):
    #black_keys_pattern = ['D#', 'F#', 'G#', 'A#', 'C#']
    num_black_keys = len(black_key_properties)
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
        diff = abs(black_key_mid_pts[i+1] - black_key_mid_pts[i])
        diffs.append((diff, i))
    print("diffs", diffs)
    diffs.sort(key = lambda x: x[0])
    big_diffs_idx1, big_diffs_idx2 = diffs[-1][1], diffs[-2][1]
    sorted_diffs_idx = sorted([big_diffs_idx1, big_diffs_idx2])
    print(sorted_diffs_idx)


    if abs(big_diffs_idx1 - big_diffs_idx2) == 2:
        if sorted_diffs_idx[0] == 0:
            pattern = ['A#', 'C#', 'D#', 'F#', 'G#']
        elif sorted_diffs_idx[0] == 1:
            pattern = ['G#', 'A#', 'C#', 'D#', 'F#']
        elif sorted_diffs_idx[0] == 2:
            pattern = ['F#', 'G#', 'A#', 'C#', 'D#']
    elif abs(big_diffs_idx1 - big_diffs_idx2) == 3:
        if sorted_diffs_idx[0] == 0:
            pattern = ['D#', 'F#', 'G#', 'A#', 'C#']
        elif sorted_diffs_idx[0] == 1:
            pattern = ['C#', 'D#', 'F#', 'G#', 'A#']
	#pattern.reverse()
    #print(pattern)
    return pattern

def detect_white_keys(img, black_key_properties, pattern):
	num_black_keys = len(black_key_properties)
	num_pattern = num_black_keys/5
	black_notes = pattern*num_pattern + pattern[:num_black_keys%5]
	print "black_notes", black_notes
	for i in range(num_black_keys):
		x,y,w,h = black_key_properties[i]
		#cv2.putText(img, black_notes[i], (x+w/2,y+h/2), 4, 0.25, (0,0,150))
		#cv2.putText(img, '#', (x+w/2,y+h/2+1), 4, 0.25, (0,0,150))

	# cv2.imshow('res',img)
	# cv2.waitKey(0)

	img_labels = np.copy(img)

	dividing_lines = []
	for i in range(num_black_keys):
		x,y,w,h = black_key_properties[i]
		pt1 = (int(x+w/2), y)
		pt2 = (int(x+w/2), img.shape[0])
		#cv2.line(img,pt1,pt2,(0,255,0),1)


		if i>0 and (black_notes[i] in ['F#', 'C#']):
			prevx, prevy, prevw, prevh = black_key_properties[i-1]
			print "green line", i, (x+w/2.+ prevx+prevw/2.)/2.
			pt1 = (int((x+w/2.+ prevx+prevw/2.)/2.), int((y+prevy)/2.))
			pt2 = (int((x+w/2.+ prevx+prevw/2.)/2.), img.shape[0])
			# cv2.line(img,pt1,pt2,(0,255,0),1)

	upper_white = []
	lower_white = []
	for i in range(1, num_black_keys):
		x,y,w,h = black_key_properties[i]
		prevx, prevy, prevw, prevh = black_key_properties[i-1]
		if black_notes[i] in ['F#', 'C#']:
			whitex = prevx+prevw
			whitey = prevy
			whitew = (x-prevx-prevw)/2
			whiteh = prevh
			upper_white.append((whitex, whitey, whitew, whiteh))
			lower_white.append((prevx+prevw/2,min(y+h,prevy+prevh), ((x+w/2)- (prevx+prevw/2))/2, img.shape[0]- min(y+h,prevy+prevh) ))
			upper_white.append((whitex+whitew, whitey, whitew, whiteh))
			lower_white.append((prevx+prevw/2+whitew,min(y+h,prevy+prevh), ((x+w/2)- (prevx+prevw/2))/2, img.shape[0]- min(y+h,prevy+prevh) ))
		else:
			whitex = prevx+prevw
			whitey = prevy
			whitew = x-prevx-prevw
			whiteh = prevh
			upper_white.append((whitex, whitey, whitew, whiteh))
			lower_white.append((prevx+prevw/2,min(y+h,prevy+prevh), (x+w/2)- (prevx+prevw/2), img.shape[0]- min(y+h,prevy+prevh) ))

	# cv2.imshow('res',img)
	# cv2.waitKey(0)

	return upper_white, lower_white

def main():
	#img = cv2.imread('keyboard-2.jpg')
	# detectKeyboard(img)
	img = cv2.imread('testimage2.jpg')
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# blurred = cv2.GaussianBlur(gray, (3,3), 0)
	# thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]

	# kernel = np.ones((3,3), np.uint8)
	# img_erosion = cv2.erode(thresh,kernel,iterations=5)

	# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# # _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
	# # sobely = cv2.Scharr(thresh,cv2.CV_64F,0,1)

	# cv2.imshow('sob',img_erosion)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()


	num_black_keys, black_key_properties = detectKeys(img)
	pattern = assign_white_keys(num_black_keys, black_key_properties)
	upper_white, lower_white = detect_white_keys(img, num_black_keys, black_key_properties, pattern)
	background_image = cv2.imread('background.jpg')

	num_white_keys = len(upper_white)
	frames_features = np.empty((0,0), int)
	for frame in frames:
		substraction = frame - background_image
		frame_features = np.empty((0,0), int)
		for i in range(num_black_keys):
			x,y,w,h = black_key_properties[i]
			black_key_posi = substraction[y:y+h, x:x+w]
			np.hstack((frame_features, black_keys_posi.flatten()))
		for i in range(num_white_keys):
			x1,y1,w1,h1 = upper_white[i]
			x2,y2,w2,h2 = lower_white[i]
			white_key_neg1 = substraction[y1:y1+h1, x1:x1+w1]
			np.hstack((frame_features, white_key_neg1.flatten()))
			white_key_neg2 = substraction[y2:y2+h2, x2:x2+w2]
			np.hstack((frame_features, white_key_neg2.flatten()))
		np.vstack((frames_features, frame_features))
	return frames_features
	print(pattern)

	return

if __name__ == '__main__':
	main()
