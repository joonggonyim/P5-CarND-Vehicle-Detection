##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./writeup_img/car_nocar_img.png
[image2]: ./writeup_img/car_hog.png
[image3]: ./writeup_img/nocar_hog.png
[image4]: ./writeup_img/search_window1.png
[image5]: ./writeup_img/search_window2.png
[image6]: ./writeup_img/search_window3.png
[image7]: ./writeup_img/search_window4.png
[image8]: ./writeup_img/example_find_car.png
[image9]: ./writeup_img/example_find_car_heatmap.png

[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the function named 'single_img_features' in lines 283-315 of the file called helper.py.

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I explored with different color space and number of orients to get the best performance for the SVC classifier. The resulting hog features as an image and also as a feature vector are below.

HOG Feature for Car Image:

![alt text][image2]

HOG Feature for No Car Image:

![alt text][image3]



####2. Explain how you settled on your final choice of HOG parameters.

I decided to only focus on the HOG features and also decided to only focus on varying the color space and orient parameters because they seemed to affect the performance the most. For the other parameters, I fixed them at

pix_per_cell = 16 </br>
cell_per_block = 2

I first tried to vary the color space and got the below results:

| Color Space | Orient | Time     | Accuracy [%] |
|-------------|--------|----------|--------------|
| YCrCb       | 16     | 2min 43s | 98.3390%     |
| RGB         | 16     | 3min 21s | 95.9178%     |
| HSV         | 16     | 2min 55s | 96.8328%     |
| LUV         | 16     | 2min 52s | 98.2404%     |
| HLS         | 16     | 2min 51s | 96.2979%     |
| YUV         | 16     | 2min 28s | 98.3108%     |

I noticed that 'YCrCb','LUV',and 'YUV' all performed similarly well, around 98% test accuracy. I chose to go with 'YCrCb' beacuse it had the highest Accuracy.

Now, I fixed the color space and varied the orient to see the effect of orient on classification accuracy

| Color Space | Orient | Time     | Accuracy [%] |
|-------------|--------|----------|--------------|
| YCrCb       | 8      | 1min 43s | 98.2545%     |
| YCrCb       | 16     | 2min     | 98.3249%     |
| YCrCb       | 24     | 2min 19s | 98.3671%     |
| YCrCb       | 32     | 2min 51s | 98.3108%     |

It looked like there is a small performance improvement with increase in the orient. I went with the best performing orient = 24

The final parameter choice for the HOG feature extraction is:

| Parmeter name  | value |
|----------------|-------|
| color_space    | YCrCb |
| orient         | 16    |
| pix_per_cell   | 16    |
| cell_per_block | 2     |


####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for training the classifier is in the function named 'train_classifier' in the notebook.
1. Imported the images
2. Flipped them with respect to the vertical to double the number of training images. 
3. I made sure all the images were in the range of [0,255] and type uint8. This process was checked on all iamges to ensure the same types were used for the feature extraction. 
4. Generated labels for images (0 - no car, 1 - car)
5. Extracted features from the images
6. Column-wise scaled the features using sklearn.preprocessing.RobustScaler
7. Split the test and training images with random shuffle
8. Trained the linear SV classifier using sklearn.svm.LinearSVC 


###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

After some experiments, I realized one size window cannot capture all the cars since they appear in different sizes depending on the distance from observing car. I decided to create small windows to search for cars far away and big windows to search cars closer. Also, in order to eliminate the chance of false positives and reduce the video processing time, I searched only the right side of each image. Below are some of the search windows

Window 1:
![alt text][image4]

Window 2:
![alt text][image5]

Window 3:
![alt text][image6]

Window 4:
![alt text][image7]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

As mentioned above, I tried different parameters to get the best resulting classifier. Below is an example result from the pipeline

![alt text][image8]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_out.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.


I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on an example image:

![alt text][image9]

Another thing I did to improve performance and prevent false negative, I inputed the heatmap from the previous frame to the next frame with some discount factor (< 1.0). The intuition behind this was that if vehicle was detected in the previous frame, this imformation should carry over to the future frames but with decreasing weight of importance. This allows the future frames' heatmap to build up more robuts heatmap against false positives and false negatives.


---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Right now, one of the obvious areas where my pipeline would fail is if a car appeared on the left or in front of me. This can be fixed simply by extending the search window at the cost of the video processing time.

The pipeline is noticeably weak when it tries to find a car that is far away. I can fine tune the search window to get a better search window for the far away cars.

The classifier seems to be giving lots of false positives. I think I can fix this by implementing a second classifier, (i.e. CNN) or even multiple classifiers to declare an image to be a car if multiple classifiers agree on it. 

