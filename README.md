# Behavioral Cloning
## Galen Ballew, 2017
---

This repository contains the scripts necessary to generate a working model for a autonomous vehicle via a simulator built in Unity3D. The repo does not contain the images used for training the data. I wrote a more in-depth article about the process on [Medium](https://medium.com/@galen.ballew/).

---
### Gathering data
The first step in the process was to gather data. The simulator includes a Training Mode which takes snapshots from virtual camera on the left, center, and right side of the car's dash. My training data consisted of **25,781*** observations, each with 3 different frames, steering angle, throttle, brake, and speed values.  

  <center>
  <figure>
  <img src="img/center_2017_02_11_11_48_20_547.jpg" alt="Sample Frame"/>
  <figcaption>Fig1. - A sample frame from the center camera.</figcaption>
  </figure>
  </center>


I gathered my data using the following protocol:
  1. 1 lap driven clockwise
  2. 1 lap driven counter-clockwise
  3. 1 lap driven in "recovery"

Driving in recovery meant that I only recorded the portion of driving when the car was on the outside of the lane and needed to re-center. Portions where the car was driving away from the center were not recorded because we do not want the model to learn that behavior.

This protocol was completed once for each of the two different tracks available in the simulator.

Once the data was collected, I took a quick look at the distribution of steering angles.

<center>
<figure>
  <img src="saved_graphs/angle_histogram.png"/>
<figcaption>Fig2. - Very high kurtosis distribution of steering angles.</figcaption>
</figure>
</center>

It's very clear that for the large majority of the time, there are very small steering adjustments being applied. This means that the model will learn to apply small steering adjustments most of the time which can be both good and bad. It's good to continue driving straight if the car is centered in the lane (and the lane is straight), but bad if the car is headed off of the road and needs to correct. It's up to the convolutional neural network to make a decision on when to enact the less common but higher valued steering angles.

---
### Bootstrapping

DNNs thrive on large datasets due to the problem of vanishing gradients. In order to increase my already large dataset, I simply flipped images on the y-axis and multiplied their steering angle by -1. This was a quick and easy way to double the size of my training set. Some additional ways to bootstrap:

  1. Perspective transform to birdseye view  
  2. Slight, random skews and warps of images  
  3. Random shadows added  

If these approaches where to be taken, it may serve to have multiple models with the final output being an ensemble method.

---
### Data Augmentation

I did little to none data augmentation as far as changing the feature space. However, with more road/driving conditions available in the simulator, it's quite possible that the model(s) would need specialized feature engineering. These are some of the possibilities I've thought of:

  1. Edge detection  
  2. Different color space(s)  
  3. Adding lane offset feature by combining from [Adv Lane Detection and Vehicle Tracking](https://github.com/galenballew/SDC-Lane-and-Vehicle-Detection-Tracking)
  4. Resizing images \* resized images could be used for transfer learning *
---
### Preprocessing

Per usual, all pixel data was normalized and standardized. This was done in the first layer of the CNN via a Lambda function. Additionally, the second layer of the CNN is a Crop2D layer, which removed the top of the frame (mostly pixels of the sky) and the bottom (included the hood of the car).

---
### CNN Training
The model architecture:

```python
model = Sequential()
model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(160, 320, 3), output_shape=(160, 320, 3)))
#amount of cropping is tuneable
model.add(Cropping2D(cropping=((75, 25), (0, 0))))

model.add(Convolution2D(24,5, activation='elu'))
model.add(MaxPooling2D()) #(2,2) pool with no stride, valid padding

model.add(Convolution2D(36,5, activation='elu'))
model.add(MaxPooling2D())

model.add(Convolution2D(48,5, activation='elu'))
model.add(MaxPooling2D())

model.add(Convolution2D(64,3, activation='elu'))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(512))
model.add(Dropout(0.5))

model.add(Dense(128))
model.add(Dense(32))
model.add(Dense(1))
```

This model architure is originally from [NVIDIA](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). However, I've modified it beyond the point of recognition by removing convolutional layers, introducing maxpooling and dropout, and changing the size of the fully connected layers. Further, I've changed activation functions to [Exponential Linear Units](https://arxiv.org/abs/1511.07289). ELU's are particularly useful for quick training in models with more than 5 layers.

For training, I used the Adam optimizer in order to converge quickly (compared to SGD) with a learning rate of `1e-03`. For the loss function, I used **Mean Squared Error** in order to penalized large deviances from the target.
---
### Results
