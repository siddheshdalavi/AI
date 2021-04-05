import pandas as pd
import numpy as np
import tensorflow as tf


#make numpy values easier to read
np.set_printoptions(precision=3,suppress=True)

#load the dataset into pandas dataframe
abalone_train=pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/abalone_train.csv",
    names=["Length", "Diameter", "Height", "Whole weight", "Shucked weight",
           "Viscera weight", "Shell weight", "Age"])
print(abalone_train.head())

abalone_features=abalone_train.copy()
abalone_labels=abalone_features.pop('Age')

abalone_features=np.array(abalone_features)
print(abalone_features)
norm=tf.keras.layers.experimental.preprocessing.Normalization()
norm.adapt(abalone_features)
norm_abalone_model=tf.keras.Sequential([
    norm,
    tf.keras.layers.Dense(64),
    tf.keras.layers.Dense(1)
])

norm_abalone_model.compile(optimizer='adam',
                           loss=tf.keras.losses.MeanSquaredError())

norm_abalone_model.fit(abalone_features, abalone_labels, epochs=10)