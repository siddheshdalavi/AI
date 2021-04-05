import tensorflow as tf
import numpy as np
import os
import zipfile
import random
import shutil

download_file=tf.keras.utils.get_file("cats_and_dogs.zip",
                                                "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip")

#print(download_file)

local_zip="C:/Users/sdalavi/.keras/datasets/cats_and_dogs.zip"
zip_ref=zipfile.ZipFile(local_zip,'r')
zip_ref.extractall("C:/Users/sdalavi/.keras/datasets")
zip_ref.close()

print(len(os.listdir("C:/Users/sdalavi/.keras/datasets/PetImages/Cat")))
print(len(os.listdir("C:/Users/sdalavi/.keras/datasets/PetImages/Dog")))

#path='C:/Users/sdalavi/.keras/datasets/PetImages'
#now make different directories testing and training
try:
    os.mkdir('C:/Users/sdalavi/.keras/datasets/PetImages/testing')
    os.mkdir('C:/Users/sdalavi/.keras/datasets/PetImages/training')
    os.mkdir('C:/Users/sdalavi/.keras/datasets/PetImages/testing/cat')
    os.mkdir('C:/Users/sdalavi/.keras/datasets/PetImages/testing/dog')
    os.mkdir('C:/Users/sdalavi/.keras/datasets/PetImages/training/cat')
    os.mkdir('C:/Users/sdalavi/.keras/datasets/PetImages/training/dog')
except OSError:
    pass

def split_data(SOURCE,TRAINING,TESTING,SPLIT_SIZE):
    dataset=[]
    for unitData in os.listdir(SOURCE):
        data=SOURCE+unitData
        if(os.path.getsize(data)>0):
            dataset.append(unitData)
        else:
            print("skipped" +unitData)

    train_data_length=int(len(dataset)*SPLIT_SIZE)
    test_data_length=int(len(dataset)-train_data_length)
    shuffled_set = random.sample(dataset, len(dataset))
    #shuffeld_set=random.sample(dataset,len(dataset))
    train_set=shuffled_set[0:train_data_length]
    test_set=shuffled_set[-test_data_length:]

    for unitData in train_set:
        this_file=SOURCE+unitData
        final_train_set=TRAINING+unitData
        shutil.copyfile(this_file,final_train_set)

    for unitData in test_set:
        this_file=SOURCE + unitData
        final_test_set=TESTING+unitData
        shutil.copyfile(this_file,final_test_set)

cat_source_dir="C:/Users/sdalavi/.keras/datasets/PetImages/Cat/"
training_cat_dir="C:/Users/sdalavi/.keras/datasets/PetImages/training/cat/"
testing_cat_dir="C:/Users/sdalavi/.keras/datasets/PetImages/testing/cat/"
dog_source_dir="C:/Users/sdalavi/.keras/datasets/PetImages/Dog/"
training_dog_dir="C:/Users/sdalavi/.keras/datasets/PetImages/training/dog/"
testing_dog_dir="C:/Users/sdalavi/.keras/datasets/PetImages/testing/dog/"

SPLIT_SIZE=.9
split_data(cat_source_dir,training_cat_dir,testing_cat_dir,SPLIT_SIZE)
split_data(dog_source_dir,training_dog_dir,testing_dog_dir,SPLIT_SIZE)


print(len(os.listdir("C:/Users/sdalavi/.keras/datasets/PetImages/training/cat/")))
print(len(os.listdir("C:/Users/sdalavi/.keras/datasets/PetImages/training/dog/")))
print(len(os.listdir("C:/Users/sdalavi/.keras/datasets/PetImages/testing/cat/")))
print(len(os.listdir("C:/Users/sdalavi/.keras/datasets/PetImages/testing/dog/")))

# model=tf.keras.models.Sequential([
#     tf.keras.layers.Conv2D(16,(3,3),activation='relu',input_shape=(150,150,3)),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(32,(3,3),activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Conv2D(64,(3,3),activation='relu'),
#     tf.keras.layers.MaxPooling2D(2,2),
#     tf.keras.layers.Flatten(),
#     tf.keras.layers.Dense(512,activation='relu'),
#     tf.keras.layers.Dense(1,activation='sigmoid')
# ])
#
# model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
#               loss=tf.keras.losses.binary_crossentropy,
#               metrics=['accuracy'])
#
# training_dir="C:/Users/sdalavi/.keras/datasets/PetImages/training/"
# train_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
# train_generator=train_datagen.flow_from_directory(training_dir,
#                                                   batch_size=10,
#                                                   class_mode='binary',
#                                                   target_size=(150,150))
#
# testing_dir="C:/Users/sdalavi/.keras/datasets/PetImages/testing/"
# test_datagen=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)
# test_generator=test_datagen.flow_from_directory(testing_dir,
#                                                 batch_size=10,
#                                                 class_mode='binary',
#                                                 target_size=(150,150))
#
# history=model.fit_generator(train_generator,
#                             epochs=2,
#                             verbose=1,
#                             validation_data=test_generator)