import tensorflow as tf
import numpy as np
import os
import zipfile
import pandas as pd

path_to_downloaded_file=tf.keras.utils.get_file("cats_and_dogs_filtered.zip",
                                                "https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip")
#print(path_to_downloaded_file)

local_zip="C:/Users/sdalavi/.keras/datasets/cats_and_dogs_filtered.zip"
zip_ref=zipfile.ZipFile(local_zip,'r')
zip_ref.extractall("C:/Users/sdalavi/.keras/datasets/cats&Dogs")
zip_ref.close()

#set base directory
base_dir="C:/Users/sdalavi/.keras/datasets/cats&Dogs/cats_and_dogs_filtered"

train_dir=os.path.join(base_dir,'train')
validation_dir=os.path.join(base_dir,'validation')

#directory with our training pictures
train_cats_dir=os.path.join(train_dir,'cats')
train_dogs_dir=os.path.join(train_dir,'dogs')

#directory with validation pictures
validation_cats_dir=os.path.join(validation_dir,'cats')
validation_dogs_dir=os.path.join(validation_dir,'dogs')

train_cat_fnames=os.listdir(train_cats_dir)
train_dog_fnames=os.listdir(train_dogs_dir)

#print(train_cat_fnames[:10])
#print(train_dog_fnames[:10])

#print(len(os.listdir(train_cats_dir)))
#print(len(os.listdir(train_dogs_dir)))
#print(len(os.listdir(validation_cats_dir)))
#print(len(os.listdir(validation_dogs_dir)))

#%matplotlib inline
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
nrows=4
ncols=4
pic_index=0

#set up matplotlib fig and size it to fit 4*4
fig=plt.gcf()
fig.set_size_inches(ncols*4,nrows*4)

pic_index+=8

next_cat_pic=[os.path.join(train_cats_dir,fname)
              for fname in train_cat_fnames[pic_index-8:pic_index]]

next_dog_pic=[os.path.join(train_dogs_dir,fname)
              for fname in train_dog_fnames[pic_index-8:pic_index]]

for i,img_path in enumerate(next_cat_pic+next_dog_pic):
    sp=plt.subplot(nrows,ncols,i+1)
    sp.axis('off')

    img=mpimg.imread(img_path)
    plt.imshow(img)

plt.show()

#model
model=tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    #now flattent he curve and feed to DNN
    tf.keras.layers.Flatten(),
    #512 neuron hidden layer
    tf.keras.layers.Dense(512,activation='relu'),
    #only 1 output neuron
    tf.keras.layers.Dense(1,activation='sigmoid')
    ])

model.summary()

#NOTE: In this case, using the RMSprop optimization algorithm is preferable to stochastic gradient descent (SGD),
# because RMSprop automates learning-rate tuning for us. (Other optimizers, such as Adam and Adagrad,
# also automatically adapt the learning rate during training, and would work equally well here.)

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
              loss=tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
test_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

train_generator=train_datagen.flow_from_directory(train_dir,
                                                  batch_size=10,
                                                  class_mode='binary',
                                                  target_size=(150,150))

validation_generator=test_datagen.flow_from_directory(validation_dir,
                                                      batch_size=10,
                                                      class_mode='binary',
                                                      target_size=(150,150))
history=model.fit(train_generator,
                  validation_data=validation_generator,
                  steps_per_epoch=100,
                  epochs=10,
                  validation_steps=150,
                  verbose=2)