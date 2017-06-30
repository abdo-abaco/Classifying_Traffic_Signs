# Classifying_Traffic_Signs
In this project we use what we learn about deep nueral networks and convolutional neural networks to classify traffic signs. Specifically, we train a model to classify traffic signs from the German Traffic Sign Dataset.


#**Traffic Sign Recognition** 



---

**Building a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/index1.png "Visualization"
[image2]: ./examples/index2.png "Visualization"
[image3]: ./examples/index3.png "Visualization"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"



### Data Set Summary & Exploration

Number of training examples = 34799

Number of validation examples = 4410

Number of testing examples = 12630

Image data shape = (32, 32, 3)

Number of classes = 43




### Visualization of the dataset.

A distribution of the different classes of the traffic signs.

Training Set

![alt text][image1]

Validation Set

![alt text][image2]

Test Set

![alt text][image3]

### Design and Test a Model Architecture

#### Preproccesing the Images via normalization

As a first step, I decided to convert the images to grayscale but then
learned the model performs better in the color scale.

I then normalized the image data because it seemed to improve performance.

I learned that CNNs are not rotation invariant so generating
rotated data can be very useful, I have not implemented this step.


### Architecture

The popular LeNet architecture was used as the training model. 


### Final model architecture including model type, layers, layer sizes, connectivity.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input Layer         		| 32x32x3 RGB image   							| 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Average pooling	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 3x3	    |    outputs 10x10x16 				   									|
| RELU					|												|
| Average pooling	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten		|       Input = 5x5x16. Output = 400  									|
| Fully Connected				|        Input = 400. Output = 200									|
| RELU					|												|
| DROPOUT					|												|
| Fully Connected				|        Input = 200. Output = 120									|
| RELU					|												|
| DROPOUT					|												|
| Fully Connected				|        Input = 120. Output = 43									|

 


### Training the Nueral Network 

EPOCHS = 30

BATCH_SIZE = 150

rate = 0.002



My final model results were:
* training set accuracy of 0.92
* validation set accuracy of 0.92
* test set accuracy of 0.91

If an iterative approach was chosen:
The first architecture used was the same LeNet model one execept with different dimensions used on the final layers.
These dimensions were fine-tuned to produce better results.


* What were some problems with the initial architecture?
The initial architecture used small dimensions to try to pick up the feautures and patters which span more pixels.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
Max pooling was replaced with average pooling. The step size doubled and demensions changed to bigger values on the final layer these were all experimental.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Dropout is used to prevent overfitting. The cost function tends to level at a local minimum and randomizing some weights can cause it to escape this local minimum so that it converges at a more global minumum.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

9 out of 10 times is very evident that the neural network has potential to be very effective.
 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


