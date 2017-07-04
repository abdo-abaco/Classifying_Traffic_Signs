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
[image4]: ./traffic_signs_data/1.jpg "Traffic Sign 1"
[image5]: ./traffic_signs_data/2.jpg "Traffic Sign 2"
[image6]: ./traffic_signs_data/3.jpg "Traffic Sign 3"
[image7]: ./traffic_signs_data/4.jpg "Traffic Sign 4"
[image8]: ./traffic_signs_data/5.jpg "Traffic Sign 5"



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


What was the first architecture that was tried and why was it chosen?

The first architecture used was the same LeNet model one execept with different dimensions used on the final layers.
These dimensions were fine-tuned to produce better results.

What were some problems with the initial architecture?
The initial architecture used small dimensions to try to pick up the feautures and patters which span more pixels.

How was the architecture adjusted and why was it adjusted?
Max pooling was replaced with average pooling. The step size doubled and demensions changed to bigger values on the final layer these were all experimental.

Which parameters were tuned? How were they adjusted and why?
The step size was slightly increase so that the training converges faster.

What are some of the important design choices and why were they chosen?
etc. ?
A few layers were chosen to capture the complex shapes of the traffic images.



### Training the Nueral Network 

EPOCHS = 30

BATCH_SIZE = 150

rate = 0.002


My final model results were:
* training set accuracy of 0.92
* validation set accuracy of 0.92
* test set accuracy of 0.91


Dropout is used to prevent overfitting. The cost function tends to level at a local minimum and randomizing some weights can cause it to escape this local minimum so that it converges at a more global minumum.

9 out of 10 times is very evident that the neural network has potential to be very effective.
 

### Test a Model on New Images

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because it is completely unknown to the neural network.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Slippery road      		| Slippery road  									| 
| General caution   			| General caution 										|
| Right-of-way at the next intersection				| Right-of-way at the next intersection											|
| Vehicles over 3.5 metric tons prohibited     		| End of no passing				 				|
| Roundabout mandatory			| Roundabout mandatory     							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares less favorably to the accuracy on much bigger test test.

### 3. Softmax probabilities for each prediction.

The softmax probabilities are around 99% certain which does not make sense because there was an incorrect value. Much more investigation is needed on this part.



