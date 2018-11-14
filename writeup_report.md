# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images/training_set_distribution.png "Training set distribution"
[image2]: ./images/validation_set_distribution.png "Validation set distribution"
[image3]: ./images/grayscale.png "Grayscale input image"
[image4]: ./data/mytest/sign3.png "Test Image 1"
[image5]: ./data/mytest/sign4.png "Test Image 2"
[image6]: ./data/mytest/sign12.png "Test Image 3"
[image7]: ./data/mytest/sign18.png "Test Image 4"
[image8]: ./data/mytest/sign34.png "Test Image 5"
[image9]: ./images/RGB.png "RGB input image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/lx-px/TrafficSignClassification/blob/master/CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the training and validation data set. The chart is a histogram showing the number of samples for each class. 

![alt text][image1]

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Since the color information doesn't help classification much as per the Sermamet paper, I tried the Y channel and grayscale converted input image. Finally I settled on the grayscale image.
Here is one example of one the images converted to grayscale.

![alt text][image9] 

![alt text][image3] 

The data is further process to center the mean at zero. I do this normalization by doing (image-128)/128. 128 because it is an unit8 input image.
The mean and the variance after normalization are:

`[mean, variance] = [-0.36002157403085455, 0.51667497963792453]`


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model I used is modified LeNet-5 model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|-----------------------|-----------------------------------------------| 
| Input         		| 32x32x1 grayscale image   					| 
| Convolution 5x5x1x6 	| 1x1x1x1 stride, same padding, outputs 28x28x6 |
| RELU					|												|
| Max pooling	      	| 1x2x2x1 stride,  outputs 14x14x6 				|
| Convolution 5x5x6x16  | 1x1x1x1 stride, same padding, outputs 10x10x16|
| Max pooling	      	| 1x2x2x1 stride,  outputs 5x5x6 				|
| Flatten				|												|
| Dropout		      	| dropout rate = 0.5 while training				|
| Fully connected		| input 400, outputs 120						|
| RELU					|												|
| Fully connected		| input 120, outputs 84  						|
| RELU					|												|
| Fully connected		| input 84, outputs 43  						|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

My training flow can be explained as below:
*For every epoch:
**For training I split the training set into validation and training set.
**I then shuffle the training set
**For every batch for the training set:
**Run training operation on every training batch created from the training set
**Calculate the validation accuracy for the segregated validation set after training all batches

I noticed a major negative turn in training accuracy after 12 epochs which is why I cut the training at 12 epochs.


After experimentation I settled with following hyperparameters:
epochs = 12
batch_size = 128
rate = 0.001
dropout rate = 0.5
optimizer = AdamOptimizer from Tensorflow
normally distributed convolution kernel init., mu  = 0
normally distributed convolution kernel init., sigma = 0.1
validation set partition = 20% of test set
training accuracy expectation > 0.93 (actual = 0.985 at epoch 12)



#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.1%
* validation set accuracy of 93.7%
* test set accuracy of 92.1%

My final model is a modified version of LeNet-5 architecture which is a well know architecture.

I tried following approaches to get the desired training accuracy:
** Using the Y channel of input channel from YUV conversion.
** Using RGB images as input to LeNet-5
** Using a version of Sermanet, by connecting subsampled first layer and convolution second layer to the first flattened layer.
** Changing the convolution kernel and flatten weith vector sizes
** Trying different dropout rates in combination to above techniques

Signs are mainly identified using standardized shapes than color which LeNet-5 is good at identifying since it very well works even on written letter shapes. 
Sermanet is also a CNN based structure which worked best on this data set. 

My training accuracy started high and remained high before adding a dropout layer but the validation frequency was below 89%. This made me thing the data is overfitted.
Adding a dropout layer shows the training frequency starting a low number but going higher towards the end of all epochs. The validation accuracy was about 93% as well consistently.
I think this shows that the model is not overfitted and good for testing set classification.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

In the current saved run all images were classified correctly but the sign for "Speed limit of 50km/hr" can be confused with "Speed limit of 80km/hr" as the softmax probabilities are close.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:


| Class ID	| Image			        |     Prediction        			| 
|-----------|-----------------------|-----------------------------------| 
|	3		| Speed limit (60km/h)  | Speed limit (50km/h)              |
|	4		| Speed limit (70km/h)  | Speed limit (70km/h) 				| 
|	12		| Priority road    		| Priority road					 	|
|	18		| General caution		| General caution					|
|	34		| Turn left ahead       | Turn left ahead 					|



The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

First Image:
3, Speed limit (60km/h)

| Probability         	|Class ID|     Prediction	        		     	         |	 
|-----------------------|--------|---------------------------------------------------|
| 3.84e-01         		|	2	 | Speed limit (50km/h)   				 	   	     | 
| 3.81e-01     			|	1	 | Slippery road 					   				 |
| 1.13e-01				|	5	 | End of no passing by vehicles over 3.5 metric tons| 
| 7.36e-02	      		|	3    | No passing for vehicles over 3.5 metric tons	 |
| 2.14e-02				|   10   | Right-of-way at the next intersection			 |


Second Image:
4, Speed limit (70km/h)

| Probability         	|Class ID|     Prediction	        		     | 
|-----------------------|--------|---------------------------------------| 
| 9.66e-01              |	4	 | Speed limit (70km/h) 				 | 
| 2.70e-02   			|	1	 | Speed limit (120km/h)				 |
| 3.05e-03				|	7	 | Speed limit (30km/h)					 | 
| 2.75e-03	    		|	5    | Stop	    		 		             |
|2.20e-04				|   0    | Speed limit (20km/h)      			 |

Third Image:
12, Priority road

| Probability         	|Class ID|     Prediction	        		     | 
|:----------------------|--------|---------------------------------------| 
| 9.99e-01        		|	12	 | Priority road   			         	 | 
| 1.37e-07     			|	2	 | Roundabout mandatory 				 |
| 2.38e-08				|	7	 | No vehicles					  		 | 
| 6.35e-09	    		|	15   | Speed limit (50km/h)		 		  	 |
| 1.13e-09				|   1    | Ahead only     			             |

Fourth Image:
18, General caution

| Probability         	|Class ID|     Prediction	        		     	   | 
|:----------------------|--------|---------------------------------------------| 
| 9.99e-01        		|	18	 | General caution                             | 
| 5.508e-06     	    |	24	 | Pedestrians 					               |
| 1.76e-06				|	26	 | Traffic signals						       | 
| 1.75e-06	    		|	27   | Road narrows on the right				   |
| 2.33e-08				|   29   | Right-of-way at the next intersection       |


Fifth Image:
34, Turn left ahead


| Probability         	|Class ID|     Prediction	        		     | 
|-----------------------|--------|---------------------------------------| 
| 9.99e-01        		|	34	 | Turn left ahead   				     | 
| 4.45e-05     			|	38	 | Keep right					         |
| 4.08e-06				|	36	 | Beware of ice/snow					 | 
| 6.09e-07	    		|	13   | Ahead only				             |
| 2.58e-07				|   35   | Right-of-way at the next intersection |



