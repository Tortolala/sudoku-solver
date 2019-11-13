import argparse
import logging
import cv2 as cv
import operator
import numpy as np
import os
import pickle
from cnn import predict

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


def infer_grid(img):
	"""Infers 81 cell grid from a square image."""
	squares = []
	side = img.shape[:1]
	side = side[0] / 9
	for i in range(9):
		for j in range(9):
			p1 = (i * side, j * side)  # Top left corner of a bounding box
			p2 = ((i + 1) * side, (j + 1) * side)  # Bottom right corner of bounding box
			squares.append((p1, p2))
	return squares

def display_rects(in_img, rects, colour=255):
	"""Displays rectangles on the image."""
	img = in_img.copy()
	for rect in rects:
		img = cv.rectangle(img, tuple(int(x) for x in rect[0]), tuple(int(x) for x in rect[1]), colour)
	show_image(img)
	return img

def show_image(img):
	"""Shows an image until any key is pressed"""
	cv.imshow('image', img)  # Display the image
	cv.waitKey(0)  # Wait for any key to be pressed (with the image window active)
	cv.destroyAllWindows()  # Close all windows

def cut_from_rect(img, rect):
	"""Cuts a rectangle from an image using the top left and bottom right points."""
	return img[int(rect[0][1]):int(rect[1][1]), int(rect[0][0]):int(rect[1][0])]

def scale_and_centre(img, size, margin=0, background=0):
	"""Scales and centres an image onto a new background square."""
	h, w = img.shape[:2]

	def centre_pad(length):
		"""Handles centering for a given length that may be odd or even."""
		if length % 2 == 0:
			side1 = int((size - length) / 2)
			side2 = side1
		else:
			side1 = int((size - length) / 2)
			side2 = side1 + 1
		return side1, side2

	def scale(r, x):
		return int(r * x)

	if h > w:
		t_pad = int(margin / 2)
		b_pad = t_pad
		ratio = (size - margin) / h
		w, h = scale(ratio, w), scale(ratio, h)
		l_pad, r_pad = centre_pad(w)
	else:
		l_pad = int(margin / 2)
		r_pad = l_pad
		ratio = (size - margin) / w
		w, h = scale(ratio, w), scale(ratio, h)
		t_pad, b_pad = centre_pad(h)

	img = cv.resize(img, (w, h))
	img = cv.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv.BORDER_CONSTANT, None, background)
	return cv.resize(img, (size, size))

def find_largest_feature(inp_img, scan_tl=None, scan_br=None):
	"""
	Uses the fact the `floodFill` function returns a bounding box of the area it filled to find the biggest
	connected pixel structure in the image. Fills this structure in white, reducing the rest to black.
	"""
	img = inp_img.copy()  # Copy the image, leaving the original untouched
	height, width = img.shape[:2]

	max_area = 0
	seed_point = (None, None)

	if scan_tl is None:
		scan_tl = [0, 0]

	if scan_br is None:
		scan_br = [width, height]

	# Loop through the image
	for x in range(scan_tl[0], scan_br[0]):
		for y in range(scan_tl[1], scan_br[1]):
			# Only operate on light or white squares
			if img.item(y, x) == 255 and x < width and y < height:  # Note that .item() appears to take input as y, x
				area = cv.floodFill(img, None, (x, y), 64)
				if area[0] > max_area:  # Gets the maximum bound area which should be the grid
					max_area = area[0]
					seed_point = (x, y)

	# Colour everything grey (compensates for features outside of our middle scanning range
	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 255 and x < width and y < height:
				cv.floodFill(img, None, (x, y), 64)

	mask = np.zeros((height + 2, width + 2), np.uint8)  # Mask that is 2 pixels bigger than the image

	# Highlight the main feature
	if all([p is not None for p in seed_point]):
		cv.floodFill(img, mask, seed_point, 255)

	top, bottom, left, right = height, 0, width, 0

	for x in range(width):
		for y in range(height):
			if img.item(y, x) == 64:  # Hide anything that isn't the main feature
				cv.floodFill(img, mask, (x, y), 0)

			# Find the bounding parameters
			if img.item(y, x) == 255:
				top = y if y < top else top
				bottom = y if y > bottom else bottom
				left = x if x < left else left
				right = x if x > right else right

	bbox = [[left, top], [right, bottom]]
	return img, np.array(bbox, dtype='float32'), seed_point

def extract_digit(img, rect, size):
	"""Extracts a digit (if one exists) from a Sudoku square."""

	digit = cut_from_rect(img, rect)  # Get the digit box from the whole square

	# Use fill feature finding to get the largest feature in middle of the box
	# Margin used to define an area in the middle we would expect to find a pixel belonging to the digit
	h, w = digit.shape[:2]
	margin = int(np.mean([h, w]) / 2.5)
	_, bbox, seed = find_largest_feature(digit, [margin, margin], [w - margin, h - margin])
	digit = cut_from_rect(digit, bbox)

	# Scale and pad the digit so that it fits a square of the digit size we're using for machine learning
	w = bbox[1][0] - bbox[0][0]
	h = bbox[1][1] - bbox[0][1]

	# Ignore any small bounding boxes
	if w > 0 and h > 0 and (w * h) > 100 and len(digit) > 0:
		return scale_and_centre(digit, size, 4)
	else:
		return np.zeros((size, size), np.uint8)

def get_digits(img, squares, size):
	"""Extracts digits from their cells and builds an array"""
	digits = []
	img = pre_process_image(img.copy())
	for square in squares:
		digits.append(extract_digit(img, square, size))
	return digits

def show_digits(digits, colour=255):
	"""Shows list of 81 extracted digits in a grid format"""
	rows = []
	with_border = [cv.copyMakeBorder(img.copy(), 1, 1, 1, 1, cv.BORDER_CONSTANT, None, colour) for img in digits]
	for i in range(9):
		row = np.concatenate(with_border[i * 9:((i + 1) * 9)], axis=1)
		rows.append(row)
	show_image(np.concatenate(rows))

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


logger.info('Getting image\'s corners...')
# logger.info('Saving corners image (phase 1)')
# cv.imwrite('result_corners.png', points)

cropped = crop_and_warp(img, corners)
# Saving partial result
# logger.info('Saving semi-cropped image (phase 1.5)')
# cv.imwrite('result.png', cropped)


# Transformed soduko board

# print(corners)
corners_transform = np.float32(corners)
corners_transform[2][0], corners_transform[2][1], corners_transform[3][0], corners_transform[3][1] = corners_transform[3][0],corners_transform[3][1], corners_transform[2][0], corners_transform[2][1]
transformed = transform_affine(img, corners_transform)
# img = cv.circle(transformed, tuple((388, 337)), 10, (0, 255, 0))
# Saving result phase 2
logger.info('Cropping image...')
# logger.info('Saving cropped and warped image (phase 2)')
# cv.imwrite('result.png', transformed)


# GETTING DIGITS
logger.info('Extracting digits from grid...')
squares = infer_grid(cropped)
# display_rects(cropped, squares)
digits = get_digits(cropped, squares, 28)
# show_digits(digits)
print(len(digits))
# print(digits[1])
# print(cv.bitwise_not(digits[1]))


# cv.namedWindow('normal', cv.WINDOW_NORMAL)
# cv.imshow('normal', digits[1])
# cv.namedWindow('inverted', cv.WINDOW_NORMAL)
# cv.imshow('inverted',cv.bitwise_not(digits[1]))
# cv.waitKey(0)
# cv.destroyAllWindows()

# Saving digits for test in notebook
# cv.imwrite('1.png', cv.bitwise_not(digits[1]))
# cv.imwrite('2.png', cv.bitwise_not(digits[4]))
# cv.imwrite('3.png', cv.bitwise_not(digits[5]))
# cv.imwrite('4.png', cv.bitwise_not(digits[7]))
# cv.imwrite('5.png', cv.bitwise_not(digits[12]))
# cv.imwrite('6.png', cv.bitwise_not(digits[15]))
# cv.imwrite('7.png', cv.bitwise_not(digits[19]))
# cv.imwrite('8.png', cv.bitwise_not(digits[41]))
# cv.imwrite('9.png', cv.bitwise_not(digits[39]))
# cv.imwrite('0.png', cv.bitwise_not(digits[73]))
# cv.imwrite('null.png', cv.bitwise_not(digits[0]))

logger.info('Recognizing digits...')
# gray = cv.bitwise_not(digits[12])
# print("Predicted: ", predict(gray))

recognized_digits = []
for digit in digits: 
	img = cv.bitwise_not(digit)
	recognized_digits.append(predict(img))

print(recognized_digits)