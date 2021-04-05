import tensorflow as tf
import os
from tensorflow.keras import layers
from tensorflow.keras import Model
download_file=tf.keras.utils.get_file("Horses&HUmans.h5",
                                     "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5")
#print(download_file)

from tensorflow.keras.applications.inception_v3 import InceptionV3
local_weight_file=download_file

pre_trained_model=InceptionV3(input_shape=(150,150,3),
                              include_top=False,
                              weights=None)

pre_trained_model.load_weights(local_weight_file)

for layer in pre_trained_model.layers:
    layer.trainable=False

#pre_trained_model.summary()

last_layer=pre_trained_model.get_layer('mixed7')
print("last layer output shape",last_layer.output_shape)
last_output=last_layer.output

#define a Callback class
class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if(logs.get('accuracy') > 0.97):
            print("acuracy has reached",logs.get('acuracy'), "\nHence cancelling the training!")

#from tensorflow.keras.optimizers import RMSprop

x=layers.Flatten()(last_output)
x=tf.keras.layers.Dense(1024,activation='relu')(x)
x=tf.keras.layers.Dropout(0.2)(x)
x=tf.keras.layers.Dense(1,activation='sigmoid')(x)

model=tf.keras.Model(pre_trained_model.input,x)

model.compile(tf.keras.optimizers.RMSprop(lr=0.0001),
              tf.keras.losses.binary_crossentropy,
              metrics=['accuracy'])

#model.summary()

#get the horse or human dataset
download_file_horse_or_human=tf.keras.utils.get_file("Horses or Humans.zip",
                                                     "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip")

#get the horses or humans validation datasets
download_file_horse_or_human_validation=tf.keras.utils.get_file("Horses or humans validation.zip",
                                                               "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip")

#print(download_file_horse_or_human)
import os
import zipfile

local_file="C:/Users/sdalavi/.keras/datasets/Horses or Humans.zip"
zip_ref=zipfile.ZipFile(local_file,'r')
zip_ref.extractall("C:/Users/sdalavi/.keras/datasets/training")
zip_ref.close()

local_file1="C:/Users/sdalavi/.keras/datasets/Horses or humans validation.zip"
zip_ref=zipfile.ZipFile(local_file1,'r')
zip_ref.extractall("C:/Users/sdalavi/.keras/datasets/validation")
zip_ref.close()

train_dir="C:/Users/sdalavi/.keras/datasets/training"
validation_dir="C:/Users/sdalavi/.keras/datasets/validation"

train_horses_dir="C:/Users/sdalavi/.keras/datasets/training/horses"
train_humans_dir="C:/Users/sdalavi/.keras/datasets/training/humans"

validation_horses_dir="C:/Users/sdalavi/.keras/datasets/validation/horses"
validation_humans_dir="C:/Users/sdalavi/.keras/datasets/validation/humans"

train_horses_fnames = os.listdir(train_horses_dir)
train_humans_fnames = os.listdir(train_humans_dir)
validation_humans_fnames = os.listdir(validation_humans_dir)
validation_horses_fnames = os.listdir(validation_horses_dir)

print(len(train_humans_fnames))
print(len(train_horses_fnames))
print(len(validation_horses_fnames))
print(len(validation_humans_fnames))

train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0,
                                                              rotation_range=40,
                                                              width_shift_range=0.2,
                                                              height_shift_range=0.2,
                                                              shear_range=0.2,
                                                              zoom_range=0.2,
                                                              horizontal_flip=True)

test_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

train_generator=train_datagen.flow_from_directory(train_dir,
                                                  batch_size=10,
                                                  class_mode='binary',
                                                  target_size=(150,150)
                                                  )
test_generator=test_datagen.flow_from_directory(validation_dir,
                                                batch_size=10,
                                                class_mode='binary',
                                                target_size=(150,150))

callbacks=myCallback
history=model.fit_generator(train_generator,
                            validation_data=test_generator,
                            steps_per_epoch=25,
                            epochs=4,
                            validation_steps=50,
                            verbose=2)

import matplotlib.pyplot as plt
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()