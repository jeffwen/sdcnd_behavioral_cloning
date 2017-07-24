# Self Driving Car Behavioral Cloning

[//]: # (Image References)
[final_gif]: ./media/final.gif "Final Run"

This project uses a convolutional neural network to clone driving behavior (from a driving simulator). The input were frames from a recorded video and the output was the steering angle. Read more about the project [here](https://github.com/jeffwen/sdcnd_behavioral_cloning/blob/master/behavioral_cloning.md). Check out the model running through the final 2 corners of the test track; the [video](https://vimeo.com/226684813) is not the best quality!

![final_gif]

Below is an example of the output driving from the model that was built using Keras with a Tensorflow backend. The model architecure was adapted from NVIDIA's _[End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316v1.pdf)_ paper. 

| Layer         		| Description    	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 80x320x3 YUV Image                 	   		| 
| Normalization     	| Normalize batch	                            |
| Convolution 5x5     	| 2x2 stride, valid padding, outputs 38x158x24 	|
| ELU activation		|												|
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 17x77x36   |
| ELU activation        |                                               |
| Convolution 5x5	    | 2x2 stride, valid padding, outputs 7x37x48    |
| ELU activation        |                                               |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 5x35x64    |
| ELU activation        |                                               |
| Convolution 3x3	    | 1x1 stride, valid padding, outputs 3x33x64    |
| ELU activation        |                                               |
| Dropout               | 0.5 keep probablility (training)              |
| Flatten               |                                               |
| Fully connected		| 3168 input, 100 output     					|
| Fully connected		| 100 input, 50 output     				     	|
| Fully connected		| 50 input, 10 output     				     	|
| Output         		| 10 input, 1 output     				     	|


