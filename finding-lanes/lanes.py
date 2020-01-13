import cv2
import numpy as np
import matplotlib.pyplot as plt

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1,y1,x2,y2])

def average_slope_intercept(image,lines):
    # Setting up empty arrays to create left and right lines
    left_fit = []
    right_fit = []
    # Goes through the lines and grabs the parameters of two points, namely the slope and intercept of a line
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters = np.polyfit((x1,x2), (y1,y2), 1)
        slope = parameters[0]
        intercept = parameters[1]
        # If slope is <0 which means it is going up from left
        if slope <0:
            left_fit.append((slope,intercept))
        # else slope is going up from right
        else:
            right_fit.append((slope,intercept))
    # Averages all of these line of best fits and converges on 1 single average to clean/optimize
    left_fit_average = np.average(left_fit,axis = 0)
    right_fit_average = np.average(right_fit,axis = 0)
    # Grabs the coordinates from the left_fit average of slope/intercept and creates cartesian coordinates
    left_line = make_coordinates(image,left_fit_average)
    right_line = make_coordinates(image,right_fit_average)
    # returns the cartesian coordinates of left_line and right_line

    return np.array([left_line,right_line])


def canny(image):
    # Turns image into gray scale . Allows for simpler and less computationally intensive methods to define edge detection
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Via Gaussian Blur, we are able to smooth out an image to reduce noise, where it those erros may lead to skewed edges
    # The average of values is set within a kernel of some matrix size. 5,5 is a good approx (optional)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    # Cabby method allows us to calculate the gradient (derivative) in all directions, and then plots it back based on low and high thresholds
    canny = cv2.Canny(blur,50,150)
    return canny

def display_lines(image, lines):
    # Creates an empty array with the same dimensions as the original image
    line_image = np.zeros_like(image)
    # We check if it even detected a line, we must check if our lines is not an empty array
    if lines is not None:
        # Filters through our Hough Transform to see where our lines were detected (not empty)
        for line in lines:
            # Each line is a 2D Array containing line coordinates in form x1,y1,x2,y2
            # We pull the data and we reshape the array into a 1D Array for processing 1 dimension with 4 elements 1x4
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1,y1), (x2,y2), (255,0,0), 5)
    return line_image


def region_of_interest(image):
    height = image.shape[0]
    polygons = np.array([[(200,height), (1100,height), (550,250)]])
    # creates an array with the same dimensions as our image
    mask = np.zeros_like(image)
    #Applying the polygon onto the mask. such that the area within will be completely white
    cv2.fillPoly(mask,polygons,255)
    # Taking the bitwise of both images as we saw earlier in the theory section takes the bitwise and of each homogolous pixel
    # in both arrays, ultimately masking the canny image to only show the region of interest traced by the polygonal contour of the masking
    masked_image = cv2.bitwise_and(image,mask)
    return masked_image

#loads in the image and allows opencv to read the image and associate a particular RGB/pixel Value
image = cv2.imread('test_image.jpg')
#Via Numpy, takes the numerical values and creates a numerical array of the image for processing
lane_image = np.copy(image)

# process_canny = canny(lane_image)
# cropped_image = region_of_interest(process_canny)
# # Hough Transform takes all of the points and via polar coordinates, finds all of the possible lines that connect each white points
# # Via a bin method, where it ranks greatest number of lines intersecting a bin as the most probable line, it will filter it towards that directions
#
# lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
# averaged_lines = average_slope_intercept(lane_image,lines)
# print (averaged_lines.shape)
# line_image = display_lines(lane_image, averaged_lines)
# combo_image = cv2.addWeighted(lane_image, 0.8, line_image, 1, 1)
# cv2.imshow('result',combo_image)
# cv2.waitKey(0)

cap = cv2.VideoCapture('test2.mp4')
while (cap.isOpened()):
    #Decodes each frame
    _, frame = cap.read()
    process_canny = canny(frame)
    cropped_image = region_of_interest(process_canny)
    # Hough Transform takes all of the points and via polar coordinates, finds all of the possible lines that connect each white points
    # Via a bin method, where it ranks greatest number of lines intersecting a bin as the most probable line, it will filter it towards that directions

    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intercept(frame,lines)
    line_image = display_lines(frame, averaged_lines)
    combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    cv2.imshow('result',combo_image)
    # To quit out of the loop, otherwise we are unable to stop the program until video finishes. Checks if our q is pressed ~ True
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
