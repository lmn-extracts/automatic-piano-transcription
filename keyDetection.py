import numpy as np
import cv2
import itertools as it

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

		    cv2.line(img,(int(x_plot_1*ratio),int(y_plot_1*ratio)), (int(x_plot_2*ratio),int(y_plot_2*ratio)),(0,255,0),2)
		    cv2.line(resized,(x_plot_1,y_plot_1),(x_plot_2,y_plot_2),(0,255,0),1)
	cv2.imshow('res',resized)
	cv2.imwrite('result.jpg', img[int(topX_cached*ratio):int(bottomX_cached*ratio),:,:])
	cv2.waitKey(0)
	cv2.destroyAllWindows()
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

def detectKeys(img):
	# Resize to 160x120 to make processing faster
	# resized = cv2.resize(img, (160,120), cv2.INTER_AREA)
	# ratio = img.shape[0] / float(resized.shape[0])

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	blurred = cv2.GaussianBlur(gray, (1,1), 0)
	thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY)[1]
	kernel = np.ones((5,5), np.uint8)
	thresh = cv2.erode(thresh,kernel,iterations=2)

	cnts = cv2.findContours(thresh.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
	print len(cnts[0])
	for c  in cnts[1]:
		shape = detectShape(c)
		print shape
		c = c.astype("float")
		c = c.astype(int)
		if shape == 'rectangle':
			cv2.drawContours(img, [c], -1, (255,0,0), 3)
			# cv2.imshow('res',img)
			# cv2.waitKey(0)


	cv2.imshow('res',thresh)
	cv2.imshow('rese',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()	
	return

def main():
	# img = cv2.imread('keyboard-2.jpg')
	# detectKeyboard(img)
	img = cv2.imread('result.jpg')
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
	detectKeys(img)
	return

if __name__ == '__main__':
	main()
