➢Vehicle and Person Detection

In my implementation, I used a Deep Learning approach to image recognition. Specifically, Convolutional Neural Networks (CNNs) to recognize images.
However, the task at hand is not just to detect a vehicle’s presence, but rather to point to its location. It turns out CNNs are suitable for these type of problems as well. 
The main idea is that since there is a binary classification problem (vehicle/person), a model could be constructed in such a way that it would have an input size of a small training sample (e.g., 64x64) and a single-feature convolutional layer of 1x1 at the top, which output could be used as a probability value for classification.
Having trained this type of a model, the input’s width and height dimensions can be expanded arbitrarily, transforming the output layer’s dimensions from 1x1 to a map with an aspect ratio approximately matching that of a new large input.
➢Data

Udacity equips students with the great resources for training the classifier. Vehicles and non-vehicles and computer generated HUMAN samples  have been used for training.
The total number of vehicle’s images used for training, validation, and testing was about 7500

In the first step, the dataset is explored. It consists of images taken from the GTI vehicle image database, the KITTI vision benchmark suite, and examples extracted from the project video itself. There are two classes, cars and non-cars. The cars have a label of 1.0, whereas the non-cars/human have a label of 0.0
There is a total number of 7000samples available, each image is colored and has a resolution of 64x64 pixels. The dataset is split into the training set (7000 samples) and validation set ( 1000+ samples). The distribution shows that the dataset is very balanced, which is important for training the neural network later. Otherwise, it would have a bias towards one of the two classes.\
![image](https://user-images.githubusercontent.com/71150528/113412410-cd808080-93d5-11eb-9397-4634a046b8b8.png)

              Samples of data set\







Model
Adding  convolutional layers, and flatten the final result to feed into the densely connected layers.

    model = tf.keras.models.Sequential([
    # Note the input shape is the desired size of the image 64x64 with 3 bytes color
    # This is the first convolution
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(64, 64, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # The second convolution
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The third convolution
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    # The fourth convolution
    #tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    #tf.keras.layers.MaxPooling2D(2,2),
    # The fifth convolution
    # Flatten the results to feed into a DNN
    tf.keras.layers.Flatten(),
    # 512 neuron hidden layer
    tf.keras.layers.Dense(512, activation='relu'),
    # Only 1 output neuron. It will contain a value from 0-1 where 0 for 1 class ('CAR') and 1 for the other ('humans')
    tf.keras.layers.Dense(1, activation='sigmoid')
])




➢Model.summary
A neural network is used as a deep-learning approach, to decide which image is a car and which is a HUMAN. The fully-convolutional network looks like this,which is shown by code :
model.summary()


Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_4 (Conv2D)            (None, 62, 62, 16)        448       
_________________________________________________________________
max_pooling2d_4 (MaxPooling2 (None, 31, 31, 16)        0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 29, 29, 32)        4640      
_________________________________________________________________
max_pooling2d_5 (MaxPooling2 (None, 14, 14, 32)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 12, 12, 64)        18496     
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 6, 6, 64)          0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2304)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 512)               1180160   
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 513       
=================================================================
Total params: 1,204,257
Trainable params: 1,204,257
Non-trainable params: 0
_________________________________________________________________







➢Training
Let's train for 15 epochs -- this may take a few minutes to run.Do note the values per epoch.
The Loss and Accuracy are a great indication of progress of training. It's making a guess as to the classification of the training data, and then measuring it against the known label, calculating the result. Accuracy is the portion of correct guesses.
history = model.fit(
      train_generator,
      steps_per_epoch=8,  
      epochs=15,
      verbose=1,validation_data = validation_generator,
      validation_steps=8)

➢Running the Model
Let's now take a look at actually running a prediction using the model. This code will allow we to choose 1 or more files from file system, it will then upload them, and run them 
import numpy as np
from google.colab import files
from keras.preprocessing import image
uploaded = files.upload()
for fn in uploaded.keys():
 
  # predicting images
  path = '/content/' + fn
  img = image.load_img(path, target_size=(64, 64))
  plt.axis('Off') # Don't show axes (or gridlines)
  plt.subplot(2,3,2)
  plt.imshow(img,cmap='gray')
  plt.show()
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)

  images = np.vstack([x])
  classes = model.predict(images, batch_size=10)
  print(classes[0])
  if classes[0]>0.5:
    print(fn + " is a human")
  else:
    print(fn + " is a car")
through the model, giving an indication of whether the object is a car or a human.
➢Output:
As we got accuracy of 90% but it's on limited data if we try on a more complex sample and the model gives the wrong answer.
![image](https://user-images.githubusercontent.com/71150528/113412426-d96c4280-93d5-11eb-9787-7da3b316499d.png)
![image](https://user-images.githubusercontent.com/71150528/113412429-dc673300-93d5-11eb-87bf-56d686222a09.png)



Visualizing Intermediate Representations
To get a feel for what kind of features our convnet has learned, one fun thing to do is to visualize how an input gets transformed as it goes through the convnet.
As we can see we go from the raw pixels of the images to increasingly abstract and compact representations. The representations downstream start highlighting what the network pays attention to, and they show fewer and fewer features being "activated"; most are set to zero. This is called "sparsity." Representation sparsity is a key feature of deep learning.
These representations carry increasingly less information about the original pixels of the image, but increasingly refined information about the class of the image. we can think of a convnet (or a deep network in general) as an information distillation pipeline.
![image](https://user-images.githubusercontent.com/71150528/113412449-f0129980-93d5-11eb-8901-45489fe1be02.png)




