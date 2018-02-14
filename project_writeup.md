# Project: Follow Me

[fcn]: ./images/fcn.png
[fcn_conv]: ./images/fcn_conv.png
[sim_crowd]: ./images/sim_crowd.png
[sim_zigzag]: ./images/sim_zigzag.png
[train]: ./images/train.png

## Introduction

The project task is to locate and follow a moving target. 
This can be done by analyzing individual camera frames coming from a front-facing camera on the drone. 
Then we need to classify each pixel of each frame using a fully convolutional neural network.

## Data Collection

Perhaps one of the most important tasks when working with a neural network is the collection and preparation of data.
A simple data set was provided in the project. Also given instructions how to collect additional data using a simulator and data collection best practices.  
All angles of hero:
![All angles of hero][sim_zigzag]

The hero in the dense crowd:
![Hero data in the dense crowd][sim_crowd]

## Network Architecture

Since our task is to obtain information about the location of the hero we will use Fully Convolutional Neural Network. In this network, all layers are convolutional layers. Fully connected layers are good for image classification tasks, but they do not preserve spatial information.  
The model will consist from:  
* Two encoder layers
* 1x1 convolution
* Two decoder layers
* Skip connections between the encoder and decoder layers

![Network Architecture][fcn_conv]

### Encoder

The Encoder part extracts features from the image. A deeper encoder, more complex shapes that it can extract.
We will use Separable Convolutions. This is a technique that reduces the number of parameters needed, thus increasing efficiency for the encoder network.  
The number of filters/feature maps for a given convolutional layer tends to be chosen based on empirical performance rather on theoretical justifications.  
It is usually more efficient to build a network deeper than wider. Therefore, if the chosen values of the number of filters did not help to achieve the desired result, I would add another layer.

```python
def encoder_block(input_layer, filters, strides):
    
    # Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    output_layer = separable_conv2d_batchnorm(input_layer, filters, strides)
    
    return output_layer
```

### 1x1 Convolution

At the output of the encoder, we have a 4-dimensional tensor. Now we need to extract features from it. We can not use fully connected layer because the output of the convolutional layer should be flattened into a 2-dimensional tensor, this leads to a loss of spatial information because no information about the location of the pixels is preserved. We can avoid that by using 1x1 convolution. 1x1 convolution helped in reducing the dimensionality of the layer. They are also very fast. Basically, they are a cheap way to make the net deeper, without adding too much computation. Also, replacement of fully-connected layers with convolutional layers presents an added advantage that during testing we can feed images of any size into our trained network. 

### Decoder

The Decoder part upscale Encoder output back to the input image dimensions. However, when we decode the output of the image back to the original image size some information can be lost. Skip connection is a way of retaining the information easily. In our case, each layer of the Decoder contains a skip connection to the corresponding encoder layer. Skip connection connects the output of one layer to the input of the other. As a result, the network is able to make more precise segmentation decision.

```python
def decoder_block(small_ip_layer, large_ip_layer, filters):
    
    # Upsample the small input layer using the bilinear_upsample() function.
    up_small_ip_layer = bilinear_upsample(small_ip_layer)
    
    # Concatenate the upsampled and large input layers using layers.concatenate
    concat_layer = layers.concatenate([up_small_ip_layer, large_ip_layer])
    
    # Add some number of separable convolution layers
    output_layer = separable_conv2d_batchnorm(concat_layer, filters)
    
    return output_layer
```

## Training

Because our task is computationally demanding and we have big training set then it is more rational to use GPU for training.
I already have a [configured](https://medium.com/google-cloud/running-jupyter-notebooks-on-gpu-on-google-cloud-d44f57d22dbd) Compute Engine with NVIDIA Tesla K80 in Google Cloud so I decided to use it.  
After some time playing with hyperparameters I stopped at the following:  
```python
learning_rate = 0.01 # Usually good start point
batch_size = 128 # Bigger - better, but there is a memory limitation 
num_epochs = 20 # It seems that the model stops to learn after this value
```  
Usually, hyperparameters are chosen empirically and then adjusted depending on the results.  
The number of epochs chosen so as to ensure that the training will no longer be produced.  
The size of the batch is the highest possible for this server to fit the memory.  
The learning rate was initially chosen as 0.01 and he showed himself well for achieving the goal.  

![Training curves][train]  

These hyperparameters and the network architecture helped to achieve the final score `0.42`, which is enough for submission.

## Future Enhancements

As usual in Deep Learning there are many ways to improve the results.  
For the task of this project I see the following ways to improve:  
1. Use more layers in Decoder and Encoder, make them deeper.
2. Use bigger dataset, for example, using data augmentation.
3. Try to reduce learning rate because validation loss curve is not optimal.
4. Try different optimizers for example Nadam.

## Conclusion

The model is trained only to identify the human in red clothes. If we need to change the target object to a car or a dog, we need to collect and pass additional data with such object classes to the model.
