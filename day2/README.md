<h1>Understanding CNN & Image Classification of Flowers</h1>

<h3>Image Classification of Flowers using CNN's</h3>
x shape (4323, 300, 300, 3)
y shape (4323, 5)


| Layer(type)       	        | Output Shape           |  Param #  |
| ------------- 		|:-------------:	 | -----:    |
| conv2d_1 (Conv2D)     	|  (None, 298, 298, 32)  |   896     |
| activation_1 (Activation)     |  (None, 298, 298, 32)  |    0      |
| max_pooling2d_1 (MaxPooling2  |  (None, 149, 149, 32)  |    0      |
| conv2d_2 (Conv2D)      	|  (None, 147, 147, 32)  |   9248    |
| activation_2 (Activation)     |  (None, 147, 147, 32)  |    0      |
| max_pooling2d_2 (MaxPooling2  |  (None, 73, 73, 32)  	 |    0      |
| conv2d_3 (Conv2D)      	|  (None, 71, 71, 64)    |   18496   |
| activation_3 (Activation)     |  (None, 71, 71, 64)    |    0      |
| max_pooling2d_3 (MaxPooling2  |  (None, 35, 35, 64)  	 |    0      |
| flatten_1 (Flatten)	        |  (None, 78400)  	 |    0      |
| dense_1 (Dense)	        |  (None, 64)  	 	 |   5017664 |
| activation_4 (Activation)     |  (None, 64)  		 |    0      |
| dropout_1 (Dropout)           |  (None, 64)  		 |    0      |
| dense_2 (Dense)   	        |  (None, 5)  		 |    325    |
| activation_5 (Activation)     |  (None, 5)  		 |    0      |

Total params: <b>5,046,629</b>
Trainable params: 5,046,629
Non-trainable params: 0
Train on 3242 samples, validate on 1081 samples

<img width=600 height=600 src='screenshots/output_2.png'/>

<img width=600 height=300 src='screenshots/output_1.png'/>

<img width=900 height=600 src='screenshots/output_3.png'/>
_________________________________________________________________

<h3>Studied Stanford's lecture on CNN's</h3>

<a href="https://www.youtube.com/watch?v=bNb2fEVKeEo&t=66s">[![IMAGE ALT TEXT HERE](https://img.youtube.com/vi/bNb2fEVKeEo/0.jpg)]</a>

Keywords: Convolutional neural networks, perceptron, neocognitron, LeNet, AlexNet, convolution, pooling, fully-connected layers

Slides: http://cs231n.stanford.edu/slides/2017/cs231n_2017_lecture5.pdf



<ol>
	<li>
<img width=600 height=300 src='screenshots/1.png'/>
<img  width=600 height=300 src='screenshots/2.png'/>
	</li>


<li>
<img width=600 height=300  src='screenshots/3.png'/>
<img width=600 height=300 src='screenshots/4.png'/>
</li>

<li>
<img width=600 height=300 src='screenshots/5.png'/>
<img width=600 height=300 src='screenshots/6.png'/>
</li>

<li>
<img width=600 height=300 src='screenshots/7.png'/>
<img width=600 height=300 src='screenshots/8.png'/>

</li>


<li>
<img width=600 height=300 src='screenshots/9.png'/>
<img width=600 height=300 src='screenshots/10.png'/>
</li>

<li>
<img width=600 height=300  src='screenshots/11.png'/>
<img width=600 height=300  src='screenshots/12.png'/>

</li>


<li>
<img width=600 height=300  src='screenshots/13.png'/>
<img width=600 height=300 src='screenshots/14.png'/>

</li>

<li>
<img width=600 height=300 src='screenshots/15.png'/>
<img width=600 height=300 src='screenshots/16.png'/>

</li>

<li>

<img width=600 height=300  src='screenshots/18.png'/>

</li>

</ol>
