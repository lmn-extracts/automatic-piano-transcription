import numpy as np
import cv2
import itertools as it
import imutils


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

'''
def detectShape(c):
	shape = "unidentified"
	peri = cv2.arcLength(c, True)
	approx = cv2.approxPolyDP(c, 0.04 * peri, True)
	print "approx", len(approx)
	if len(approx) > 4:
		(x,y,w,h) = cv2.boundingRect(approx)
		ar = w / float(h)

		shape = "square" if ar>=0.95 and ar<=1.05 else "rectangle"
	return shape
'''

def detect_black_keys(img, show=False):
	# Resize to 160x120 to make processing faster
	# resized = cv2.resize(img, (160,120), cv2.INTER_AREA)
	# ratio = img.shape[0] / float(resized.shape[0])
	print "img", img.shape
	image = img[10:,:].copy()

	# convert the resized image to grayscale, blur it slightly,
	# and threshold it
	# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# blurred = cv2.GaussianBlur(gray, (1,1), 0)
	# thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
	# kernel = np.ones((5,5), np.uint8)
	# thresh = cv2.erode(thresh,kernel,iterations=2)

	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)[1]
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]

	external_black_key_widths = []
	for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
		# shape = detectShape(c)
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
		# shape = detectShape(c)
		# if shape == "rectangle":

		# multiply the contour (x, y)-coordinates by the resize ratio,
		# then draw the contours and the name of the shape on the image
		c = c.astype("int")
		# cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
		# cv2.fillPoly(image, [c], (0, 255, 0))
		x,y,w,h = cv2.boundingRect(c)
		all_list_black_key_widths.append(w)

	print "external_black_key_widths",external_black_key_widths
	print "all_list_black_key_widths", all_list_black_key_widths
	black_key_widths = external_black_key_widths + all_list_black_key_widths

	print "black_key_widths" , black_key_widths
	best_black_key_width = np.bincount(black_key_widths).argmax()
	black_key_properties = []
	for c in cnts:
	# compute the center of the contour, then detect the name of the
	# shape using only the contour
		# shape = detectShape(c)
		# if shape == "rectangle":
		# multiply the contour (x, y)-coordinates by the resize ratio,
		# then draw the contours and the name of the shape on the image
		c = c.astype("int")
		x,y,w,h = cv2.boundingRect(c)
		if abs(w - best_black_key_width) <= 1:
			black_key_properties.append((x,y, w,h))


	#default black keys is detected from right to left, we reverse to starting from left before output it
	black_key_properties.sort(key = lambda x: x[0])
	# if show:
	# 	cv2.imwrite(str(ind) + '.jpg', image)
	# 	ind+=1
	return black_key_properties

def assign_black_keys(black_key_properties):
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
    #print("diffs", diffs)
    diffs.sort(key = lambda x: x[0])
    big_diffs_idx1, big_diffs_idx2 = diffs[-1][1], diffs[-2][1]
    sorted_diffs_idx = sorted([big_diffs_idx1, big_diffs_idx2])
    print(sorted_diffs_idx)

    black_keys_pattern = []
    if abs(big_diffs_idx1 - big_diffs_idx2) == 2:
        if sorted_diffs_idx[0] == 0:
            black_keys_pattern = ['A#', 'C#', 'D#', 'F#', 'G#']
        elif sorted_diffs_idx[0] == 1:
            black_keys_pattern = ['G#', 'A#', 'C#', 'D#', 'F#']
        elif sorted_diffs_idx[0] == 2:
            black_keys_pattern = ['F#', 'G#', 'A#', 'C#', 'D#']
    elif abs(big_diffs_idx1 - big_diffs_idx2) == 3:
        if sorted_diffs_idx[0] == 0:
            black_keys_pattern = ['D#', 'F#', 'G#', 'A#', 'C#']
        elif sorted_diffs_idx[0] == 1:
            black_keys_pattern = ['C#', 'D#', 'F#', 'G#', 'A#']
    print("black_keys_pattern", black_keys_pattern)
    num_pattern = num_black_keys/5
    black_notes = black_keys_pattern*num_pattern + black_keys_pattern[:num_black_keys%5]

    return black_keys_pattern, black_notes

def detect_white_keys(img, black_key_properties, black_keys_pattern, black_notes):
	if not black_keys_pattern:
		return [], []
	num_black_keys = len(black_key_properties)
	#print "black_notes", black_notes
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

	upper_white_properties = []
	lower_white_properties = []
	for i in range(1, num_black_keys):
		x,y,w,h = black_key_properties[i]
		prevx, prevy, prevw, prevh = black_key_properties[i-1]
		if black_notes[i] in ['F#', 'C#']:
			whitex = prevx+prevw
			whitey = prevy
			whitew = (x-prevx-prevw)/2
			whiteh = prevh
			upper_white_properties.append((whitex, whitey, whitew, whiteh))
			lower_white_properties.append((prevx+prevw/2,min(y+h,prevy+prevh), ((x+w/2)- (prevx+prevw/2))/2, img.shape[0]- min(y+h,prevy+prevh) ))
			upper_white_properties.append((whitex+whitew, whitey, whitew, whiteh))
			lower_white_properties.append((prevx+prevw/2+whitew,min(y+h,prevy+prevh), ((x+w/2)- (prevx+prevw/2))/2, img.shape[0]- min(y+h,prevy+prevh) ))
		else:
			whitex = prevx+prevw
			whitey = prevy
			whitew = x-prevx-prevw
			whiteh = prevh
			upper_white_properties.append((whitex, whitey, whitew, whiteh))
			lower_white_properties.append((prevx+prevw/2,min(y+h,prevy+prevh), (x+w/2)- (prevx+prevw/2), img.shape[0]- min(y+h,prevy+prevh) ))

	# cv2.imshow('res',img)
	# cv2.waitKey(0)

	return upper_white_properties, lower_white_properties

def assign_white_keys(black_keys_pattern, white_properties):
	num_white_keys = len(white_properties)
	white_keys_pattern_right_of_black = []
	white_notes = []

	first_white_key_right_of_black = black_keys_pattern[0]
	if first_white_key_right_of_black == 'C#':
		white_keys_pattern_right_of_black = ['D', 'E', 'F', 'G', 'A', 'B', 'C']
	elif first_white_key_right_of_black == 'D#':
		white_keys_pattern_right_of_black = ['E', 'F', 'G', 'A', 'B', 'C', 'D']
	elif first_white_key_right_of_black == 'F#':
		white_keys_pattern_right_of_black = ['G', 'A', 'B', 'C', 'D', 'E', 'F']
	elif first_white_key_right_of_black == 'G#':
		white_keys_pattern_right_of_black = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
	elif first_white_key_right_of_black == 'A#':
		white_keys_pattern_right_of_black = ['B', 'C', 'D', 'E', 'F', 'G', 'A']

	if white_keys_pattern_right_of_black:
		num_pattern = num_white_keys/7
		white_notes = white_keys_pattern_right_of_black*num_pattern + white_keys_pattern_right_of_black[:num_white_keys%7]
	return white_keys_pattern_right_of_black, white_notes


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

	background_image = cv2.imread('background.jpg')

	black_key_properties = detect_black_keys(background_image)
	black_keys_pattern, black_notes = assign_black_keys(black_key_properties)
	upper_white_properties, lower_white_properties = detect_white_keys(background_image, black_key_properties, black_keys_pattern, black_notes)
	white_keys_pattern_right_of_black, white_notes = assign_white_keys(black_keys_pattern, upper_white_properties)
	# num_white_keys = len(upper_white_properties)
	# frames_features = np.empty((0,0), int)
	# for frame in frames:
	# 	substraction = frame - background_image
	# 	frame_features = np.empty((0,0), int)
	# 	for i in range(num_black_keys):
	# 		x,y,w,h = black_key_properties[i]
	# 		black_key_posi = substraction[y:y+h, x:x+w]
	# 		np.hstack((frame_features, black_keys_posi.flatten()))
	# 	for i in range(num_white_keys):
	# 		x1,y1,w1,h1 = upper_white_properties[i]
	# 		x2,y2,w2,h2 = lower_white_properties[i]
	# 		white_key_neg1 = substraction[y1:y1+h1, x1:x1+w1]
	# 		np.hstack((frame_features, white_key_neg1.flatten()))
	# 		white_key_neg2 = substraction[y2:y2+h2, x2:x2+w2]
	# 		np.hstack((frame_features, white_key_neg2.flatten()))
	# 	np.vstack((frames_features, frame_features))
	# return frames_features
	print(white_notes)

	return

if __name__ == '__main__':
	main()
