# -*- coding: utf-8 -*-
"""
Created on Sat Jun 23 09:55:24 2018

@author: kputtur
"""

import numpy as np
import cv2
import glob
#import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm
#from moviepy.editor import VideoFileClip
#from IPython.display import HTML

def camera_calibration(images, nx, ny):
    
    #create objp np.array with nx*ny items of type float32, 9x6 = 54 items of [0. 0. 0.]
    objp = np.zeros((ny*nx, 3), np.float32)
    
    #create a grid from [0,0]...[5,4]... [8,5]
    objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
    
    #Arrays to store objpoints and imgpoints
    objpoints = []  # 3d points in real world space
    imgpoints = []  # 2d points in image plane
   
    # Step through the image list and search for chess board corners
    for fname in tqdm(images):
        img = cv2.imread(fname)
        img_size = (img.shape[1], img.shape[0])
        #since it is cv2.imread color format will be in BGR and not RGB
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        #find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)
        
        #If corners are found add object points and image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
        
    # once we have objpoints and imgpoints we can now calibrate using the cv2.calibrateCamera function
    # Which returns the camera matrix(mtx), distortion coefficients(dist), rotation(rvecs) and translation vectors(tvecs)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size,None, None)
        
    return mtx, dist

    
images = glob.glob("camera_cal/calibration*.jpg")
mtx, dist = camera_calibration(images, 9, 6)    

# Caculate the Region of interest using trial & Error method.
left_bottom = (150, 672)
left_top = (580, 450)
right_bottom = (1200, 672)
right_top = (730, 450)
roi_points = [[left_top, right_top, right_bottom, left_bottom]]

def bounding_box(img, roi_points):
    mask = np.zeros_like(img)
    vertices = np.array(roi_points, dtype=np.int32)

    if len(img.shape) > 2:
        channel_count = img.shape[2]  
        ignore_mask_color = (255,) * channel_count
        #print(ignore_mask_color)
    else:
        ignore_mask_color = 255
        
    #The function fillPoly fills an area bounded by several polygonal contours
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    roi_image = cv2.bitwise_and(img, mask)
    return roi_image

def image_unwarp(img, roi_points):
    src = np.float32(roi_points)
    dst = np.float32([[0, 0], [640, 0], [640, 720], [0, 720]])
    # Given src and dst points, calculate the perspective transform matrix
    M  = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, (640,720), flags=cv2.INTER_LINEAR)
    return warped
  
def birds_eye(img):
    src = np.float32(roi_points)
    dst = np.float32([[0, 0], [640, 0], [640, 720], [0, 720]])
    M = cv2.getPerspectiveTransform(src, dst)
    invMtx = cv2.getPerspectiveTransform(dst, src)
    warped = cv2.warpPerspective(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
    return warped, M, invMtx
    

def abs_sobel_thresh(img, orient, sobel_kernel, thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # GaussianBlur filter will reduce or removes noise while keeping edges relatively sharp.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    
    if (orient=='x'):
        sobel = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    else:
        sobel = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    abs_sobel = np.absolute(sobel)
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return binary

def mag_thresh(img, sobel_kernel, mag_thresh=(0,255)):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    sobel = np.sqrt(sobelx**2 + sobely**2)
    scaled_sobel = np.uint8(255 * sobel / np.max(sobel))
    binary = np.zeros_like(scaled_sobel)
    binary[(scaled_sobel >= mag_thresh[0]) & (scaled_sobel <= mag_thresh[1])] = 1
    return binary
  
def dir_threshold(img, sobel_kernel, thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sobelx = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    gradir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary = np.zeros_like(gradir)
    binary[(gradir >= thresh[0]) & (gradir <= thresh[1])] = 1
    return binary

def color_threshold(img, channel='rgb', thresh=(220,255)):
    #Convert the image to RGB.
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    temp_img = img
    
    if channel is 'hls':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    elif channel is 'hsv':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    elif channel is 'yuv':    
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    elif channel is 'ycrcb':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    elif channel is 'lab':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
    elif channel is 'luv':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2Luv)
    
    img_ch1 = temp_img[:,:,0]
    img_ch2 = temp_img[:,:,1]
    img_ch3 = temp_img[:,:,2]

    bin_ch1 = np.zeros_like(img_ch1)
    bin_ch2 = np.zeros_like(img_ch2)
    bin_ch3 = np.zeros_like(img_ch3)

    bin_ch1[(img_ch1 > thresh[0]) & (img_ch1 <= thresh[1])] = 1
    bin_ch2[(img_ch2 > thresh[0]) & (img_ch2 <= thresh[1])] = 1
    bin_ch3[(img_ch3 > thresh[0]) & (img_ch3 <= thresh[1])] = 1
    
    return bin_ch1, bin_ch2, bin_ch3

def combined_color(img):
     bin_rgb_ch1, bin_rgb_ch2, bin_rgb_ch3 = color_threshold(img, channel='rgb', thresh=(230,255))
     bin_hsv_ch1, bin_hsv_ch2, bin_hsv_ch3 = color_threshold(img, channel='hsv', thresh=(230,255))    
     bin_luv_ch1, bin_luv_ch2, bin_luv_ch3 = color_threshold(img, channel='luv', thresh=(157,255))

     binary = np.zeros_like(bin_rgb_ch1)
    
     binary[(bin_rgb_ch1 == 1) | (bin_hsv_ch3 == 1) | (bin_luv_ch3 == 1) ] = 1
    
     return binary


def process_image(img, plot=False):
    # Distortion correction
    undist = cv2.undistort(img, mtx, dist, None, mtx)
    # Birds eye perspective transform
    warped, M, invMtx = birds_eye(undist)
    # Color thresholding
    binary = combined_color(warped)
    return binary, undist, warped, invMtx
    
# Input: binary_img
# Output: return left lane, right lane, left lane pixel indices, right lane pixel indices 
def find_lane_lines(binary_img, prev_left_fit=None, prev_right_fit=None, friction=0., plot=False):
    # Set the width of the windows +/- margin
    margin = 100
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Start sliding window search if either lane not found
    if  prev_left_fit is None or np.all(prev_left_fit == 0) or prev_right_fit is None or np.all(prev_right_fit == 0):
        # Take a histogram of the bottom half of the image
        histogram = np.sum(binary_img[binary_img.shape[0]//2:,:], axis=0)
        # Find the peak of the left and right halves of the histogram
        # These will be the starting point for the left and right lines
        midpoint = np.int(histogram.shape[0]/2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_img.shape[0]/nwindows)
        # Current positions to be updated for each window
        leftx_current = leftx_base
        rightx_current = rightx_base
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        out_img = np.dstack((binary_img, binary_img, binary_img))*255
        
        
        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_img.shape[0] - (window+1)*window_height
            win_y_high = binary_img.shape[0] - window*window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high), (0,255,0), 2) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high), (0,255,0), 2) 
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                             (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
                              (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:        
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        
        
        
        
    else:     # Reuse lanes from last frame
        left_lane_inds = ((nonzerox > (prev_left_fit[0]*(nonzeroy**2) + prev_left_fit[1]*nonzeroy + 
                           prev_left_fit[2] - margin)) & (nonzerox < (prev_left_fit[0]*(nonzeroy**2) + 
                           prev_left_fit[1]*nonzeroy + prev_left_fit[2] + margin))) 

        right_lane_inds = ((nonzerox > (prev_right_fit[0]*(nonzeroy**2) + prev_right_fit[1]*nonzeroy + 
                            prev_right_fit[2] - margin)) & (nonzerox < (prev_right_fit[0]*(nonzeroy**2) + 
                            prev_right_fit[1]*nonzeroy + prev_right_fit[2] + margin)))
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    if len(leftx) > 0 and len(lefty) > 0:
        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        
        if plot:
            ploty = np.linspace(0, binary_img.shape[0]-1, binary_img.shape[0] )
            left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
            right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]

            plt.figure(figsize=(10, 10))
            out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
            out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
            plt.imshow(out_img)
            plt.plot(left_fitx, ploty, color='yellow')
            plt.plot(right_fitx, ploty, color='yellow')
            plt.xlim(0, 1280)
            plt.ylim(720, 0)
            plt.title('Lane finding and curve fitting')
            plt.show()
    else:
        # Couldn't find lanes
        return None, None, left_lane_inds, right_lane_inds
    
    # Use momentum update
    if prev_left_fit is not None and prev_right_fit is not None:
        left_fit = friction * prev_left_fit + (1.0 - friction) * left_fit
        right_fit = friction * prev_right_fit + (1.0 - friction) * right_fit

    return left_fit, right_fit, left_lane_inds, right_lane_inds
  
  
  
# Draw left and right curves on original imagee. Fill space between lane cuves with color.
# Input: binary_img, left lane, right lane, left lane indices, right lane indices
# Output: image with lane filled, curve radius, center offset
def draw_lane_lines(img, binary_img, left_fit, invMtx, right_fit, left_lane_inds, right_lane_inds, plot=False):
    # Set the width of the windows +/- margin
    #margin = 100
    curve_radii = None
    center_offset = None
    
    ploty = np.linspace(0, binary_img.shape[0]-1, binary_img.shape[0])
    left_fitx = left_fit[0]* ploty**2 + left_fit[1]*ploty + left_fit[2]
    right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(binary_img).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))
    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, invMtx, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    lane_filled_img = cv2.addWeighted(img, 1, newwarp, 0.3, 0)

    # Calculate curve radius    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = binary_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Define conversions in x and y from pixels space to meters
    y_eval = np.max(ploty)
    ym_per_pix = 30 / 720 # meters per pixel in y dimension
    xm_per_pix = 3.7 / 700 # meters per pixel in x dimension
    
    if len(leftx) > 0 and len(lefty) > 0 and len(rightx) > 0 and len(righty) > 0:
        # Fit new polynomials to x,y in world space
        left_fit_cr = np.polyfit(lefty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(righty * ym_per_pix, rightx * xm_per_pix, 2)
 # Calculate the new radii of curvature
        left_curve_radius = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2 * left_fit_cr[0])
        right_curve_radius = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2 * right_fit_cr[0])    
        curve_radii = (left_curve_radius, right_curve_radius)

        # Calculate center offset
        w, h = img.shape[1], img.shape[0]
        left_point = left_fit[0] * h**2 + left_fit[1] * h + left_fit[2]
        right_point = right_fit[0] * h**2 + right_fit[1] * h + right_fit[2]
        midpoint = (right_point - left_point) * 0.5 + left_point
        center_offset = (midpoint - w * 0.5) * xm_per_pix
    
    if plot:
        print('Left lane R: {:.2f}m Right lane R: {:.2f}m'.format(left_curve_radius, right_curve_radius))
        print('Offset from center: {:.2f}m'.format(center_offset))
        
        f, ax = plt.subplots(1, 2, figsize=(20, 20))
        ax[0].imshow(img)
        ax[0].set_title('Original image')
        ax[1].imshow(lane_filled_img)
        ax[1].set_title('Lane filled image')
        
        plt.show()
    
    return curve_radii, center_offset, lane_filled_img

# path list of test images
test_images = glob.glob('./test_images/straight*.jpg')
for test_img_path in test_images:
    test_img = mpimg.imread(test_img_path)
    print(test_img_path)
    binary, undist, warped, invMtx = process_image(test_img)
    left_fit, right_fit, left_lane_inds, right_lane_inds = find_lane_lines(binary, plot=True)
    #curve_radii, center_offset, lane_filled_img = draw_lane_lines(undist, binary, invMtx, left_fit, right_fit, left_lane_inds, right_lane_inds, plot=True)
    plt.show()  
  