# Udacity Self Driving Car Engineer Nanodegree Project 4

##  Advanced Lane Finding

## Writeup Template  

### I will finish and submit the writeup as a markdown file, as I usually do in my earlier projects.

---

**Advanced Lane Finding Project**

The goal of this project is to develop a pipeline to process a video stream from a forward-facing camera mounted on the front of a car, and output an(or several) annotated video which identifies:

* step1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* step2: Apply a distortion correction to raw images.
* step3: Use color transforms, gradients, etc., to create a thresholded binary image.
* step4: Apply a perspective transform to rectify binary image ("birds-eye view").
* step5: Detect lane pixels and fit to find the lane boundary.
* step6: Determine the curvature of the lane and vehicle position with respect to center.
* step7: Warp the detected lane boundaries back onto the original image.
* step8: Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image and video References)

[image1]: ./output_images/calibration2.jpg "calibration1"
[image2]: ./output_images/calibration14.jpg "calibration2"
[image3]: ./output_images/undistorted_straight_lines1.jpg "Road Transformed"
[image4]: ./output_images/undistorted_test4.jpg "Road Transformed"
[image5]: ./output_images/undistorted_straight_lines2.jpg "Road Transformed"
[image6]: ./output_images/undistorted_test5.jpg "Road Transformed"
[image7]: ./output_images/undistorted_and_warped_straight_lines2.jpg "straightlines2"
[image8]: ./output_images/undistorted_and_warped_test5.jpg
[image9]: ./output_images/combined_thresholded_straight_lines2.jpg "straightlines2"
[image10]: ./output_images/combined_thresholded_test5.jpg
[image11]: ./examples/curve_fitting.jpg "curve fitting"
[image12]: ./examples/curvature.jpg "curvature"
[image13]: ./examples/curvature1.jpg
[image14]: ./output_images/fill_lanelines_straight_lines2.jpg "Fit Visual"
[image15]: ./output_images/fill_lanelines_test5.jpg "Output"


[video1]: ./project_result.mp4 "project result"
[video2]: ./challenge_result.mp4 "challenge result"
[video3]: ./harder_challenge_result.mp4 "harder challenge result"


## [Project Rubric](https://review.udacity.com/#!/rubrics/571/view) 

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup that includes all the rubric points and how you addressed each one.  As in my earlier projects, I will submit this writeup as a markdown file. 

---

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

This part includes the section named "Step 1: Camera Calibration given a set of chessboard images" in "P4.ipynb". After loading several uselful packages in the 1st code cell of P4.ipynb, the code for implementing camera calibration is shown in the 2nd code cell of my attached IPython notebook located in "P4.ipynb".

First, the object points are 3D (x, y, z) coordinates of the chessboard corners in real-world scenes, and the image points are 2D (x,y) coordinates of the corresponding chessboard corners in image plane. Assuming the chessboard is fixed on the (x, y) plane with z=0, the object points are the same for each calibration image.  Thus `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it each time I successfully detect all chessboard corners in a test image. `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

Therefore, the corners can be detected by calling opencv helper functions `cv2.findChessboardCorners()` and be drawn in the original input image by using `cv2.drawChessboardCorners()`. Here are two examples of given calibration images as follows. And more results are stored in the folder named output_images, please refer to them.


calibration2.jpg |![alt text][image1]
-----------------|------------------
calibration14.jpg|![alt text][image2]

**Please pay attention: when I run this step in practice, there are some interesting things:**

- In folder named camera_cal there are 20 given images. However, I only did camera calibration with 17 images successfully and the results are saved in output_images folder. After analyzing the failed original 'calibration1.jpg','calibration4.jpg' and 'calibration5.jpg', they have the same problem: the distortions are too severe so cv2.findChessboardCorners() can't detect enough (9,6)corners in x and y directions. Thus there are only 17 successful detected images stored in output_images folder.
- There are two ways to save the images with detected corners: cv2.imwrite()and matplotlib.figure.Figure.figsave(), and I chose the latter because fig.figsave()can save the original image and 'detected-corners' image as a whole image file once, while cv2.imwrite()can only save one 'detected-corners' image once.

Then I used the output `objpoints` and `imgpoints` to compute the camera calibration matrix and distortion coefficients using the `cv2.calibrateCamera()` in the third code cell in P4.ipnb. I applied this distortion correction to the test image using the `cv2.undistort()`. Here are two of eight test images after undistortion. And more results are included in the folder named output_images, please refer to them.

straight_lines1.jpg|![alt text][image3]
-------------------|-------------------
test4.jpg          |![alt text][image4]

---

### Pipeline (single images)

**Please pay attention:** As the [udacity courses](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/44732d48-dcfe-4b4e-9614-12422ec29306) says above, the pipeline for advanced laneline finding includes eight steps:

- step1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- step2: Apply a distortion correction to raw images.
- step3: Use color transforms, gradients, etc., to create a thresholded binary image.
- step4: Apply a perspective transform(warping) to rectify binary image ("birds-eye view").
- step5: Detect lane pixels and fit to find the lane boundary.
- step6: Determine the curvature of the lane and vehicle position with respect to center.
- step7: Warp the detected lane boudaries back into original image
- step8: Output visual display of the lane boundaries and numerical estimationof lane curvature and vehicle position.

However, after lots of trials and errors, instead of implementing color/gradient thresholds on the whole images and then warping the regions of interest(yellow and white lanelines), I think step 3 and step 4 can be exchanged optionally. And there are mainly two reasons to do this:

- 1.from the view of image processing, after perspective transform, the output images are easier and more focused on the regions of interest(yellow and white laneline regions), which is helpful for further detection and curvature calculation.
- 2.from the view of gradient thresholding, after perspecive transform, the highlighted yellow and white lanelines are almost vertical in the warped images, which is helpful for determining the gradient threshold, instead of lots of experiments.

Therefore, **my pipeline for single image/frames is as follows**, The experiments have proved my thoughts are right about the exchanging of step 3 and 4. And the results are satisfying.

- step1: Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
- step2: Apply a distortion correction to raw images.
- step3(original step4): Apply a perspective transform(warping) to undistorted image ("birds-eye view").
- step4(original step3): Use multi-channel color transforms, gradients, etc., to create a combined-thresholded binary image to warped images.
- step5: Detect lane pixels and fit to find the lane boundary.
- step6: Determine the curvature of the lane and vehicle position with respect to center.
- step7: Warp the detected lane boudaries back into original image
- step8: Output visual display of the lane boundaries and numerical estimationof lane curvature and vehicle position.

---

#### 1. Provide an example of a distortion-corrected image.

This part includes the section named "Step 2: Distortion Correction on raw images" in "P4.ipynb". And the corresponding source code is in the third and forth code cells of P4.ipynb. Here the distortion correction on raw images can be realized by building a function named undistort() which uses the calculated camera calibration matrix and distortion coefficients. And the eight undistorted images are saved in output_images folder, too. 
```sh
def undistort(filename, show = True, read = True):
    if read:
        img = cv2.imread(filename)
        
    img_size = (img.shape[1],img.shape[0])
    ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(objpoints,imgpoints,img_size,None,None)
    undist = cv2.undistort(img,mtx,dist,None,mtx)
    
    if show:
        fig,axes = plt.subplots(1,2,figsize =(8,4))
        axes[0].imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        axes[0].set_title('original image',fontsize = 12)
        axes[1].imshow(cv2.cvtColor(undist,cv2.COLOR_BGR2RGB))
        axes[1].set_title('undistorted image', fontsize = 12)
        fig.savefig(os.path.join('output_images','undistorted_'+os.path.split(filename)[1]))
    else:
        return undist    
```

Here are another two of eight test images after undistortion. And more results are included in the folder named output_images, please refer to them.

straight_lines2.jpg|![alt text][image5]
-------------------|-------------------
test5.jpg          |![alt text][image6]

**Please pay attention:**
Comparing the original images and the corresponding undistorted images, especially in the edge regions of camera, you can see the differences produced by radial and translational cameara distortion. And in real world we usually ignore these distortions. 

---

#### 2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

This part includes the section named "Step 3(original step4): Applying a perspective transform(warping) to undistorted images" in P4.ipynb. And the corresponding source code is in the 5th-6th code cells of P4.ipynb. The perspective transform,i.e.warping can be implemented by constructing a function named `birds_eye_view()`. Although perspective transform can be done between undistorted images and warped images through any possible viewpoints, a bird's-eye view of the road can help to show the lanelines almost in parallel to each other, which is very useful for further processing: combined color/gradient thresholding, laneline detection and curvature calculation.  

The `birds_eye_view` calls opencv functions `cv2.getPerspectiveTransform(src,dst)` and `cv2.warpPerspective(undist,M,img_size)` to implement warping. The source and destination points are determined in the test image named straight_lines1.jpg. And the corresponding codes are as follows:

```python
src = np.float32([[490,482], [810,482],[1250,720],[40,720]])
dst = np.float32([[0,0],[1280,0],[1280,720],[40,720]])
    
M = cv2.getPerspectiveTransform(src,dst)
warped = cv2.warpPerspective(undist,M,img_size)
```

Here are two examples of undistored test images after warping. And more results are included in the folder named output_images, please refer to them.

straight_lines2.jpg|![alt text][image7]
-------------------|-------------------
test5.jpg          |![alt text][image8]

---

#### 3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

This part includes the section named "Step 4(original step3): Implementing Color/Gradient thresholds to create a combined thresholded binary image" in P4.ipynb. And the corresponding source code is in the 7th-8th code cells of P4.ipynb.

This combined thresholding on color and gradients can be realized by building a function named combined_thresholds().

** color thresholds**

Aside from RGB and HLS colorspaces, courses encourage us to try more different colorspaces like HSV, Lab and LUV, which are mostly inspired by human vision system and/or developed for efficient use in television screen displays and computer graphics. And I did. After trials and errors, I chose the following color channels and thresholds on given test images to create binary color-thresholded images, with highlighted yellow and white lanelines.

* s-channel from HLS colorspace, with threshold [180,255], stays fairly consistent in shadow or excessive brightness for both yellow and white lanelines. However, I observed it cannot detect all the pixels of interest in either of them, so more color channel should be chose to compensate this.
* b-channel from Lab colorspace, with threshold [155,200], works better than S-channel in detecting yellow lanelins, while completely misses the white ones.
* l-channel from LUV colorspace, with threshold [225,255], works better than S-channel in detecting white lanelines,while completely misses the yellow ones.

** gradient thresholds**

Besides the color-thresholded images, I chose sobel filters on s-channel to detect edges of lanelines. After lots of trials and errors, the magnitude of gradient along x and y directions with threshold [50,150] was useful for test images. And the requirement for determining gradient threshold is to detect the most salient regions with the largest gradient without bringing extra noise to the combined color binary images. 

Based on these four channels, here comes the combined thresholded binary images as follows.

Here are two examples of warped test images (straight_lines2.jpg and test5.jpg) after combined thresholding. And more results are included in the folder named output_images, please refer to them.

![alt text][image9]
![alt text][image10]

And the kernel codes are as follows:
```python
    s_threshold = [180,255]
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_threshold[0])&(s_channel <= s_threshold[1])]=1
    
    b_threshold = [155,200]
    b_binary = np.zeros_like(b_channel)
    b_binary[(b_channel >= b_threshold[0])&(b_channel <= b_threshold[1])]=1
    
    l_threshold = [225,255]
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_threshold[0])&(l_channel <= l_threshold[1])]=1
    
    color_binary = np.zeros_like(s_binary)
    color_binary[(b_binary == 1)|(l_binary ==1)]=1
    
    sobelx = cv2.Sobel(s_channel,cv2.CV_64F,1,0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    sobel_threshold = [50,150]
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel>=sobel_threshold[0])&(scaled_sobel<=sobel_threshold[1])]=1    
    
    # get the final combined thresholded binary image
    combined_binary = np.zeros_like(grad_binary)
    combined_binary[(color_binary ==1)|(grad_binary==1)]=1
```

---

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

This part includes the section named "Step 5: Lane pixels detection and boundary fitting" in P4.ipynb. And the corresponding source code is in the 9th code cells of P4.ipynb.

From now on I can use the combined-thresholded binary image to detect yellow and white laneline pixels, and fit a polynomial curve for each boundary of them. The driveable region between the two detected lanelines is highlighted with green area.

The laneline detection can be implemented by locating peaks in the histogram of half combined-thresholded binary image, and by detecting nonzero pixels near the peaks by calling `np.nonzero()`. Using the x and y pixel positions to fit a second order polynomial curve by calling `np.polyfit()`:

![alt text][image11]

And the corresponding codes are as follows:
```python
...
x,y = np.nonzero(np.transpose(combined_binary))
...
left_fit = np.polyfit(lefty,leftx,2)
left_fitx = left_fit[0]*lefty**2 + left_fit[1]*lefty + left_fit[2]
right_fit = np.polyfit(righty,rightx,2)
right_fitx = right_fit[0]*righty**2 + right_fit[1]*righty + right_fit[2]
...
```

---

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

This part includes the section named "Step 6: Calculating lane curvatures and vehicle locations" in P4.ipynb. And the corresponding source code is in the 9th code cells of P4.ipynb.

In this section, after locating the laneline pixels, the laneline curvatures is caluculated according to [course slides](https://classroom.udacity.com/nanodegrees/nd013/parts/fbf77062-5703-404e-b60c-95b78b2f3f9e/modules/2b62a1c3-e151-4a0e-b6b6-e424fa46ceab/lessons/40ec78ee-fb7c-4b53-94a8-028c5c60b858/concepts/2f928913-21f6-4611-9055-01744acc344f) and [curvature formula](http://www.intmath.com/applications-differentiation/8-radius-curvature.php). And the equations for radius of curvature are:

|formula1 |formula2|
|-------------------|--------------------|
|![alt text][image12]| ![alt text][image13]|

The y values of my image increase from top to bottom, so when measuring the radius of curvature closest to the vehicle, the formula above at the y value should be evaluated corresponding to image bottom. Besides, the position of vehicles can be calculated by taking the average of the x intercepts of each line.

Besides, the correspongding codes are as follows:
```python
    # calculate the curvature radius for each laneline
    # meters per pixel in y dimension
    ym_per_pix = 30./720 
    # meters per pixel in x dimension
    xm_per_pix = 3.7/700
    
    left_fit_cr = np.polyfit(lefty*ym_per_pix, leftx*xm_per_pix,2)
    right_fit_cr = np.polyfit(righty*ym_per_pix, rightx*xm_per_pix,2)
    left_curverad = ((1+(2*left_fit_cr[0]*np.max(lefty) + left_fit_cr[1])**2)**1.5)/np.absolute(2*left_fit_cr[0])
    right_curvead = ((1+(2*right_fit_cr[0]*np.max(lefty) + right_fit_cr[1])**2)**1.5)/np.absolute(2*right_fit_cr[0])
    
    # calculate the position of vehicle
    center = abs(640 - ((rightx_int + leftx_int)/2))
```

---

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

This part includes the sections named "Step7: Warping detected lane boundaries back to original images"  and "Step8: Visualizing the advanced lanline finding" in P4.ipynb. And the corresponding source code is in the 9th-10th code cells of P4.ipynb.

This warping can be implemented by executing the inverse of perspective transform matrx M with exchanging the postions of source points and destination points:

```sh
offset = 0
    img_size = (combined_binary.shape[1], combined_binary.shape[0])
    src = np.float32([[490, 482],[810, 482],
                      [1250, 720],[40, 720]])
    dst = np.float32([[0, 0], [1280, 0], 
                     [1250, 720],[40, 720]])
    Minv = cv2.getPerspectiveTransform(dst, src)
    
    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
    pts_left = np.array([np.flipud(np.transpose(np.vstack([left_fitx, lefty])))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx,righty]))])
    pts = np.hstack((pts_left,pts_right))
    cv2.polylines(color_warp, np.int_([pts]), isClosed = False, color = (0,0,255), thickness = 40)
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255,0))
    newwarp = cv2.warpPerspective(color_warp, Minv,img_size)
    result = cv2.addWeighted(mpimg.imread(filename), 1, newwarp, 0.5, 0)
```

Then the cv2.warpPerspective()realizes the transform from detected lane boundaries back to the original undistorted images.

All of these step 4,5 and 6 can be implemented by building a function named fill_lanelines(). Here are two examples of warped test images (straight_lines2.jpg and test5.jpg) after inverse-warping. And more results are included in the folder named output_images, please refer to them.

straight_lines2.jpg|![alt text][image14]
-------------------|-------------------
test5.jpg          |![alt text][image15]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).
This part includes the corresponding source code is in the 11th-18th code cells of P4.ipynb.

In this section, firstly I define a class named Line() as described in udacity courses, where the laneline attributes can be stored from frame to frame. And inside the class, some functions are built to do advanced laneline detection.

Then I built a pipeline function named video_processing(), which will do the similar pipeline fill_lanelines() for frame to frame, instead of a single image. The video_processing includes camera calibration, distortion correction, perspective transform, combined thresholding, laneline detection, curvature calculation and so on. Besides, the difference between these two functions is video_processing() will store information of the lanelines across frames to average the lane positions and produce a robust smooth ouput.

The video_processing() also knows whether the lanelines were detected in the previous frame, and if detected, it only checks for laneline pixels in a tight window around the previous polynomial, ensuring a high confidence detection. And if not detected in the previous frame(or the first five frames), the function will do the same search which was perforemed in fill_lanelines()based on identifying left and right peaks in a histogram of combined-thresholded binary undistored image.

Here's a [link to the project_video result](./project_result.mp4), and the result of project_video.mp4 saved as project_result.mp4.

The corresponding codes are as follows:
```python
Left = Line()
Right = Line()
video_output = 'project_result.mp4'

clip = VideoFileClip('project_video.mp4')
white_clip = clip.fl_image(video_processing)
white_clip.write_videofile(video_output,audio = False)
```

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?
#### Optional test videos named 'challenge_video.mp4' and 'harder_challenge_video.mp4'
Here's a [link to the challenge_video result](./challenge_result.mp4),and the result of challenge_video.mp4 saved as challenge_result.mp4. 

Now you can see the processing on challenge_video.mp4 is satisfying, because the highway shadow and illumination aren't not severe so my algorithm piple can handle it. 

Here's a [link to the harder challenge_video result](./harder_challenge_result.mp4),and the result of challenge_video.mp4 saved as harder_challenge_result.mp4.

However, when processing harder_challenge_video.mp4, where the environment is much more scarttered and distracted, and the shadows change severely, so my pipeline cannot perform well. So in further work, I will focus on how to improve the core module like combined_thresholded_binary() and fill_lanelines(). 