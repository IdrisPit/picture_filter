import cv2
import numpy as np

def dummy(val):
	pass
	
identity_kernel = np.array([[0,0,0],[0,1,0],[0,0,0]])
sharpen_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
gaussian_kernel1 = cv2.getGaussianKernel(3, 0)
gaussian_kernel2 = cv2.getGaussianKernel(5, 0)
box_kernel = np.array([[1,1,1],[1,1,1],[1,1,1]], np.float32) / 9

kernels = [identity_kernel, sharpen_kernel, gaussian_kernel1, gaussian_kernel2, box_kernel]
	
color_original = cv2.imread('test.jpg')
W = 600
height, width, depth = color_original.shape
imgScale = W/width
newX, newY = color_original.shape[1]*imgScale, color_original.shape[0]*imgScale
rs_color_original = cv2.resize(color_original, (int(newX), int(newY)))

color_modified = rs_color_original.copy()

gray_original = cv2.cvtColor(rs_color_original, cv2.COLOR_BGR2GRAY)
gray_modified = gray_original.copy()

cv2.namedWindow('Image Editor')
cv2.createTrackbar('contrast', 'Image Editor', 1, 100, dummy)
cv2.createTrackbar('brightness', 'Image Editor', 50, 100, dummy)
cv2.createTrackbar('filter', 'Image Editor', 0, len(kernels)-1, dummy)
cv2.createTrackbar('grayscale', 'Image Editor', 0, 1, dummy)

count = 1 #counter for images saved

while True:
	grayscale = cv2.getTrackbarPos('grayscale', 'Image Editor')
	if grayscale == 0:
		cv2.imshow('Image Editor', color_modified)
	else:
		cv2.imshow('Image Editor', gray_modified)

	k = cv2.waitKey(1) & 0xFF
	if k == ord('q'):
		break
	elif k == ord('s'):
		if grayscale == 0:
			cv2.imwrite('output%d.png' % count, color_modified)
		else:
			cv2.imwrite('output%d.png' % count, gray_modified)
		count = count + 1
		
	contrast = cv2.getTrackbarPos('contrast', 'Image Editor')
	brightness = cv2.getTrackbarPos('brightness', 'Image Editor')
	kernel = cv2.getTrackbarPos('filter', 'Image Editor')
	
	color_modified = cv2.filter2D(rs_color_original, -1, kernels[kernel])
	gray_modified = cv2.filter2D(gray_original, -1, kernels[kernel])
	
	color_modified = cv2.addWeighted(color_modified, contrast, np.zeros(rs_color_original.shape, dtype=rs_color_original.dtype), 0, brightness-50)
	gray_modified = cv2.addWeighted(gray_modified, contrast, np.zeros(gray_original.shape, dtype=gray_original.dtype), 0, brightness-50)
	
cv2.destroyAllWindows()

