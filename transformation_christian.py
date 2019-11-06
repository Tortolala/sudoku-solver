# !/bin/python
""" Select 4 points on image with mouse events and apply affine transform
"""
import cv2 as cv
import numpy as np 


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


def mouse_coordinates(event, x, y, flags, params):
    """ Left button mouse click event function
    Args: None

    Returns: Appends coordinate tuple to global list
    """
    global click_points
    if event == cv.EVENT_LBUTTONDOWN:
        print("{0}\t{1}".format(x,y))
        click_points.append((x, y))


def draw(im_original):
    """ Window to select 4 points in an image
    """
    win_name = "Select four points"
    cv.namedWindow(win_name)
    cv.setMouseCallback(win_name, mouse_coordinates)
    print('X\tY')
    while True:
        im = im_original.copy()
        for point in click_points:
            cv.circle(im, point, 2, (0, 255, 0), -1)
    
        cv.imshow(win_name, im)

        # quit loop with esc key
        if (cv.waitKey(1) == 27) | (len(click_points) == 4):
            for point in click_points:
                cv.circle(im, point, 2, (0, 255, 0), -1)
            cv.imshow(win_name, im)
            break

if __name__=='__main__':
    import argparse
    click_points = []
    parser = argparse.ArgumentParser(prog='Handling mouse events')
    parser.add_argument('-i', type=str, default=r"/Users/pq5xttu8/Desktop/Tortolala/Repositories/sudoku-solver/sudoku.jpg")  #win
    args = parser.parse_args()

    im_original = cv.imread(args.i)
    draw(im_original)

    transformed = None
    # perform transform only if all corners have been defined
    if (len(click_points) == 4):
        
        print(click_points)
        pts1 = np.float32(click_points)
        print(pts1)
        transformed = transform_affine(im_original, pts1)
        
        win_name = "Affine transform"
        cv.namedWindow(win_name)
        cv.imshow(win_name, transformed)
        cv.waitKey(0)
            

# [(170, 208), (789, 182), (251, 508), (677, 504)]  <- clickpoints
# [[170. 208.]  
#  [789. 182.]
#  [251. 508.] 
#  [677. 504.]]  <- 