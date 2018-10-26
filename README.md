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
[image4]: ./data/mytest/00028.png "Test Image 1"
[image5]: ./data/mytest/00029.png "Test Image 2"
[image6]: ./data/mytest/00030.png "Test Image 3"
[image7]: ./data/mytest/00031.png "Test Image 4"
[image8]: ./data/mytest/00032.png "Test Image 5"

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

![alt text][image3]

The data is further process to center the mean at zero. I do this normalization by doing (image-128)/128. 128 because it is an unit8 input image.
The mean and the variance after normalization are:

[mean, variance] = [-0.36002157403085455, 0.51667497963792453]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model I used is modified LeNet-5 model.

My final model consisted of the following layers:
|:---------------------:|:---------------------------------------------:|
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
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
|:---------------------:|:---------------------------------------------:|

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
* training set accuracy of 0.985
* validation set accuracy of 0.949
* test set accuracy of 0.921

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

|:----------|:---------------------:|:---------------------------------:| 
| Class ID	| Image			        |     Prediction        			| 
|:----------|:---------------------:|:---------------------------------:| 
|	2		| Speed limit (50km/h)  | Speed limit (50km/h)              | 
|	4		| Speed limit (70km/h)  | Speed limit (70km/h) 				|
|	4		| Speed limit (70km/h)  | Speed limit (70km/h) 				|
|	38		| Keep right			| Keep right						|
|	12		| Priority road    		| Priority road					 	|
|:----------|:---------------------:|:---------------------------------:| 

The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. 

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

First Image:
2, Speed limit (50km/h)

|:---------------------:|:------:|--------------------------------------------:|
| Probability         	|Class ID|     Prediction	        		     	   |	 
|:---------------------:|:------:|--------------------------------------------:|
| 0.99         			|	2	 | Speed limit (50km/h)   				 	   | 
| 1.0e-04     			|	5	 | Speed limit (80km/h) 					   |
| 6.9e-06				|	3	 | Speed limit (60km/h)						   | 
| 5.3e-06	      		|	1    | Speed limit (20km/h)					 	   |
| 5.5e-10				|   10   | No passing for vehicles over 3.5 metric tons|
|:---------------------:|:------:|--------------------------------------------:|

Second Image:
4, Speed limit (70km/h)

|:---------------------:|:------:|--------------------------------------:|
| Probability         	|Class ID|     Prediction	        		     | 
|:---------------------:|:------:|--------------------------------------:| 
| 9.9e-01               |	38	 | Keep right  							 | 
| 5.3e-07     			|	34	 | Turn left ahead						 |
| 2.3e-07				|	36	 | Go straight or right					 | 
| 1.9e-09	    		|	1    | Speed limit (20km/h)			 		 |
| 2.4e-10				|   12   | Priority road      					 |
|:---------------------:|:------:|--------------------------------------:|

Third Image:
4, Speed limit (70km/h)

|:---------------------:|:------:|--------------------------------------:|
| Probability         	|Class ID|     Prediction	        		     | 
|:---------------------:|:------:|--------------------------------------:| 
| 9.9e-01        		|	4	 | Speed limit (70km/h)   				 | 
| 2.6e-05     			|	0	 | Speed limit (20km/h) 				 |
| 2.5e-05				|	1	 | Speed limit (30km/h)					 | 
| 1.8e-05	    		|	8    | Speed limit (120km/h)		 		 |
| 1.1e-07				|   7    | Speed limit (100km/h)      			 |
|:---------------------:|:------:|--------------------------------------:|

Fourth Image:
38, Keep right

|:---------------------:|:------:|--------------------------------------------:|
| Probability         	|Class ID|     Prediction	        		     	   | 
|:---------------------:|:------:|--------------------------------------------:| 
| 9.9e-01        		|	12	 | No passing for vehicles over 3.5 metric tons| 
| 1.8e-03     			|	40	 | Roundabout mandatory 					   |
| 1.3e-03				|	41	 | End of no passing						   | 
| 8.9e-04	    		|	13   | Yield					 		 		   |
| 3.5e-04				|   11   | Right-of-way at the next intersection       |
|:---------------------:|:------:|--------------------------------------------:|

Fifth Image:
12, Priority road

|:---------------------:|:------:|--------------------------------------:|
| Probability         	|Class ID|     Prediction	        		     | 
|:---------------------:|:------:|--------------------------------------:| 
| 8.9e-01        		|	4	 | Speed limit (70km/h)   				 | 
| 7.4e-02     			|	0	 | Speed limit (20km/h)					 |
| 2.9e-02				|	1	 | Speed limit (30km/h)					 | 
| 2.3e-03	    		|	8    | Speed limit (120km/h)				 |
| 6.7e-04				|   14   | Stop      					  		 |
|:---------------------:|:------:|--------------------------------------:|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


