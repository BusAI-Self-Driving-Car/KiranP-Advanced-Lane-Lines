## Adavanced Lane Lines Project - Term 1

### In this project, my goal is to write a software pipeline to identify the lane boundaries in a video from a front-facing camera on a car. 

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./assets/distort_chess.png "Distort Chessboard"
[image2]: ./assets/road_distort.png "Road Transformed"
[image3]: ./assets/warped.png       "Warped"
[image4]: ./assets/bounding_box.png "Region of Interest"
[image5]: ./assets/rewarp_road.png  "Road Rewarped"
[image6]: ./assets/binary_img.png   " Binary Output"
[image7]: ./assets/magnitude.png    " Magnitude"
[image8]: ./assets/binary_direction.png   " Binary Direction"
[image9]: ./assets/combined_color_binary.png   " Combined Color Binary"
[image10]: ./assets/color_space_RGB.png   " Color Space:RGB"
[image11]: ./assets/color_space_HLS.png   " Color Space:HLS"
[image12]: ./assets/color_space_HSV.png   " Color Space:HSV"
[image13]: ./assets/chosen_binary.png     " Combined Color Binary"
[image14]: ./assets/histogram_1           " Histogram"
[image15]: ./assets/sliding_windows.png   " Sliding Windows"
[image16]: ./assets/polynomial_fit.png    " Polynomial Fit"
[image17]: ./assets/tuning_windows.png    " Tuning Windows"
[image18]: ./assets/measuring_curvature.png   " Measuring Curvature"
[image19]: ./assets/final_result.png      " Final Color Binary"
[image20]:  ./assets/mtransform.png       "mtransform"
[video1]: ./output_project_video_1.mp4    "Video"

## Final Video of the Advanced Lane Mapping

![youtubeimage](http://img.youtube.com/vi/x8CfUd0MzoM/maxresdefault.jpg) 

[youtube_link](https://youtu.be/x8CfUd0MzoM "youtube_link")

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---


### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the first code cell of the IPython notebook located in "./Advanced_Lane_lines.ipynb" from 3rd Cell
function ```python def camera_calibration(images, nx, ny):```

Here the images for camera calibration are taken from folder `camera_cal` using glob function we will read all the images assuming all images belongs to chess board. Once the images are read we can make use of the function camera calibration function to calibrate the camera.

`def camera_calibration` creates objp numpy array with 9x6 = 54 points starts from 0,0... 8,5. Loop over the images, convert the images to Gray using cv2.cvtColor function. Now we need to find the corners in the 9x6 board using the cv2.findChessboardCorners function. Once the call for the above is successful we append 3d points in objpoints & imgpoints in image plane.

once we have objpoints and imgpoints we can now calibrate using the cv2.calibrateCamera function Which returns the camera matrix(mtx), distortion coefficients(dist), rotation(rvecs) and translation vectors(tvecs)
More information about the cv2 function is here.
[OpenCV Docs](https://docs.opencv.org/3.4/dc/dbb/tutorial_py_calibration.html)


![mtransform][image20] 

I start by preparing "object points", which will be the (9, 6, 3) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (9, 6) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (9, 6) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![distort Chess board][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:
Here I have taken the test images from the test_images folder and Chessboard from `camera_cal` folder.
![Road View Distortion][image2]
![Chessboard Distortion][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image. 
Look at the following python functions about the color spaces I explored.

| Function name                              | Description                    |
| ------------------------------------------ | ------------------------------ |
| `abs_sobel_thresh`                         | Sobel Filter.                  |
| `mag_thresh`                               | Magnitude of Gradient !        |
| `Combined Gradient`                        | Magnitude, Sobel, Direction    |
| `rgb_thresh`                               | Color Space :RGB               |
| `hls_thresh`                               | Color Space :HLS               |
| `hsv_thresh`                               | Color Space : HSV!             |
| `combined_color`                           | ***Combined Color + Binary***  |

![Sobel][image6]
![Magnitude][image7]
![Binary Direction][image8]
![Color and Binary][image9]
![Color Space RGB][image10]
![Color Space HLS][image11] 
![Color Space HSV][image12] 


As you can see after trying out all the different color spaces such as RGB, HLS, HSV I settled for Combined color code is as shown below

```python
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
```
Which tries to convert to HLS, HSV, YUV, YCRB, LAB & LUV color spaces and then combining them like this

```python
def combined_color(img):
     bin_rgb_ch1, bin_rgb_ch2, bin_rgb_ch3 = color_threshold(img, channel='rgb', thresh=(230,255))
     bin_hsv_ch1, bin_hsv_ch2, bin_hsv_ch3 = color_threshold(img, channel='hsv', thresh=(230,255))    
     bin_luv_ch1, bin_luv_ch2, bin_luv_ch3 = color_threshold(img, channel='luv', thresh=(157,255))

     binary = np.zeros_like(bin_rgb_ch1)
    
     binary[(bin_rgb_ch1 == 1) | (bin_hsv_ch3 == 1) | (bin_luv_ch3 == 1) ] = 1
    
     return binary
```
![Final Chosen Color Space][image13]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

After doing some trial and error with image sizes and trying to draw the Region of Interest and the perspective transform points I decided to manually set it so that it is easier to modify following is the source and destination of the perspective Transform:

```python
left_bottom = (150, 672)
left_top = (580, 450)
right_bottom = (1200, 672)
right_top = (730, 450)
roi_points = [[left_top, right_top, right_bottom, left_bottom]]

dst = np.float32([[0, 0], [640, 0], [640, 720], [0, 720]])
    
```
![Bounding Box/ ROI][image4]
![Rewarping the Road][image5]

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 150, 672      | 0  , 0        | 
| 580, 450      | 640, 0        |
| 1200, 672     | 640, 720      |
| 730, 450      | 0,   720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After applying calibration, thresholding, and a perspective transform to a road image, I take a binary image where the lane lines stand out clearly. However,I still need to decide explicitly which pixels are part of the lines and which belong to the left line and which belong to the right line.

I first take a histogram along all the columns in the lower half of the image like this:

```python
import numpy as np
import matplotlib.pyplot as plt

histogram = np.sum(img[img.shape[0]//2:,:], axis=0)
plt.plot(histogram)```

![Histogram][image14]


##### Sliding Windows

With this histogram I am adding up the pixel values along each column in the image. In my thresholded binary image, pixels are either 0 or 1, so the two most prominent peaks in this histogram will be good indicators of the x-position of the base of the lane lines. I can use that as a starting point for where to search for the lines. From that point, I can use a sliding window, placed around the line centers, to find and follow the lines up to the top of the frame.
![Sliding Windows][image15]

#### Ploynomial Fit

Polynomial Least Squares comprises a broad range of statistical methods for estimating an underlying polynomial that describes observations.
![numpy polyfit](https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.polyfit.html)

My implementation can be found in the function
```python
def polynomial_fit(warped, left_indices, right_indices, left_fit, right_fit):
```
![Polynomial Fit][image16]

##### Tuning the Windows once we have polynomial fit
```python
def modified_windows(warped, left_fit, right_fit):
```

![Tuned Windows][image17]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

Calculation of Radius of Curvature of Lane is extensively discussed in the Class Materials.The radius of curvature are calculate by following equation:

R = [1 + (2Ay + B)^2]^3/2 / |2A| 



the postion to center is the distance between middle of image(640) and the middel of lane lines at the bottom.

![Radius of Curvature][image18]

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

You can run the in[29] cell in the ipynb notebook and findout the implementation.  The steps are as follows

1. Read an image, undistort, find out Region of Interest, get the Warped version using `image_unwarp` and then used the `combined_binary` 

2. Find out the left & right indices using the `sliding_window` function

3. Get the polynomial fit and get the `left_poly` and `right_poly`

4. Calculate the lane using `calculate_lane`

5. Find the curvature using `curvature_distance`

6. Draw the lane using the `draw_lane`

7. Draw the Final image using the `draw_final`


![Final Image][image19]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a ![local file](./output_project_video_1.mp4)

![youtube_link](http://img.youtube.com/vi/x8CfUd0MzoM/maxresdefault.jpg "youtube_link")



youtube link for the same is here ![youtube Link](https://www.youtube.com/watch?v=x8CfUd0MzoM, "youtube link")


---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

**1. Finding the Suitable Color Spaces for Lane Recognition**
   I initially struggled to find suitable color spaces and binary combination for the lane recognition, I tried RGB, HSV, LAB, YCRB all independently. I spent most of time trying to find the right color + binary combination but each of the color spaces had one or other problems, whichever showed clear marking of lane in one test image would eventually wear out during next test image. So I did explore other possibilies I saw in one of the slack channels, student mentioned combining the color, which is what I did eventually.

**2. Shadows Causing lane recognition to fail at some places**
    In the harder challenges there were lot of shadows around so I thought of exploring this to see how we can detect lanes in shadows. I read little bit about it one of the option is to use the CLAHE histogram equalization which I used during my project, this is known to give
better hold of the shadow environment. This is something in my todo list

**3. Bumpy Roads causing the Lane recognition to go Haywire**
  I saw in some places my lane detection went haywire, this happened when the Roadsurface suddenly changed or I guess vehicle hit a rough patch suddenly jumped? I can't tell But this is something that needs to be handled !! Again one more in my todo list.


**4. Visibility/Weather and other conditions.**
  I didn't really factor in the weather, Visibility and other conditions, which can definately have more impact, Color of the road surface something which is not widely discussed. How does the Car behave on mud roads ? how can we detect lanes if there is lot of sunshine etc.
  
**5. ROI, Curvature, Lane detection Tuning**
 Calculating the ROI is handcoded to the point, which I think should be dervied from the image sizes rather. Lane detection tuning took lot of time, trying to evaluate whether we have best fitting polynomial or is there a deviation etc.

