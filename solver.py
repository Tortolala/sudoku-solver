import argparse
import logging
import cv2 as cv
import operator
import numpy as np

# Basic config
logging.basicConfig(level = logging.INFO)
logger = logging.getLogger('sudoku-solver')

parser = argparse.ArgumentParser(description='Sudoku solver')
parser.add_argument('--p', type=str, required = True, help='path to image to evaluate')

image_path = parser.parse_args().p
logger.info(f"Getting image from {image_path}")

# Functions definitions
def pre_process_image(img):
	"""Uses a blurring function, adaptive thresholding and dilation to expose the main features of an image."""

	proc = cv.GaussianBlur(img.copy(), (9, 9), 0)
	proc = cv.adaptiveThreshold(proc, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)
	proc = cv.bitwise_not(proc, proc)

	return proc

def display_points(in_img, points, radius=5, colour=(0, 255, 0)):
	"""Draws circular points on an image."""
	img = in_img.copy()

	if len(colour) == 3:
		if len(img.shape) == 2:
			img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
		elif img.shape[2] == 1:
			img = cv.cvtColor(img, cv.COLOR_GRAY2BGR)

	for point in points:
		img = cv.circle(img, tuple(int(x) for x in point), radius, colour, -1)

	return img

def find_corners_of_largest_polygon(img):
	"""Finds the 4 extreme corners of the largest contour in the image."""
	contours, h = cv.findContours(img.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)  
	contours = sorted(contours, key=cv.contourArea, reverse=True) 
	polygon = contours[0]  # Largest image

	bottom_right, _ = max(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_left, _ = min(enumerate([pt[0][0] + pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	bottom_left, _ = min(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))
	top_right, _ = max(enumerate([pt[0][0] - pt[0][1] for pt in polygon]), key=operator.itemgetter(1))

	return [polygon[top_left][0], polygon[top_right][0], polygon[bottom_right][0], polygon[bottom_left][0]]

def distance_between(p1, p2):
	"""Returns the scalar distance between two points"""
	a = p2[0] - p1[0]
	b = p2[1] - p1[1]
	return np.sqrt((a ** 2) + (b ** 2))

def crop_and_warp(img, crop_rect):
	"""Crops and warps a rectangular section from an image into a square of similar size."""

	top_left, top_right, bottom_right, bottom_left = crop_rect[0], crop_rect[1], crop_rect[2], crop_rect[3]
	src = np.array([top_left, top_right, bottom_right, bottom_left], dtype='float32')

	side = max([
		distance_between(bottom_right, top_right),
		distance_between(top_left, bottom_left),
		distance_between(bottom_right, bottom_left),
		distance_between(top_left, top_right)
	])

	dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')
	m = cv.getPerspectiveTransform(src, dst)

	return cv.warpPerspective(img, m, (int(side), int(side)))


# Transform fucntion by Christian
def transform_affine(im, pts1):
    """ Apply affine transform to image given a set of reference points.
    Args:
        im (numpy array): Source image
        pts (list): 4 tuples of (x,y) coordinates of corners 

            
    Returns:
        transformed (numpy array): Affine transformed image from selected points.

    Reference points must be provided in the following order for perspective rectification:
    pts = [top-left ,top-rifgt, bottom-left, bottom-right]

                pt 0           pt 1
                    ------------
                    |          |
                    |          |
                    |          |
                    |          |
                    ------------
                pt  2          pt 3
    """

    rows = 512
    cols = 512
    
    pts2 = np.float32([[0,0],[rows,0],[0,cols],[rows,cols]])
    
    # compute transformation matrix
    m = cv.getPerspectiveTransform(pts1,pts2)
    # apply affine transform
    transformed = cv.warpPerspective(im, m,(rows,cols))

    return transformed





# Reading image 
original_img = cv.imread(image_path)
img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

# Main processing
processed = pre_process_image(img)

ext_contours, hier = cv.findContours(processed.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
contours, hier = cv.findContours(processed.copy(), cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
all_contours = cv.drawContours(processed.copy(), contours, -1, (0, 255, 0), 2)
external_only = cv.drawContours(processed.copy(), ext_contours, -1, (0, 255, 0), 2)

corners = find_corners_of_largest_polygon(processed)
points = display_points(processed, corners)


logger.info('Saving corners image (phase 1)')
cv.imwrite('result_corners.png', points)

cropped = crop_and_warp(img, corners)
# Saving partial result
# logger.info('Saving semi-cropped image (phase 1.5)')
# cv.imwrite('result.png', cropped)


# Transformed soduko board

# print(corners)
corners_transform = np.float32(corners)
corners_transform[2][0], corners_transform[2][1], corners_transform[3][0], corners_transform[3][1] = corners_transform[3][0], corners_transform[3][1], corners_transform[2][0], corners_transform[2][1]
transformed = transform_affine(img, corners_transform)
# img = cv.circle(transformed, tuple((388, 337)), 10, (0, 255, 0))
# Saving result phase 2
logger.info('Saving cropped and warped image (phase 2)')
cv.imwrite('result.png', transformed)
