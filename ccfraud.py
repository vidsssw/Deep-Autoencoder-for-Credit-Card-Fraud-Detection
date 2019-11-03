import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import numpy as np
import random
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import regularizers

def prepoc_data(df):
    scaler=StandardScaler()
    df['Time']=scaler.fit_transform(df[['Time']])
    df['Amount']=scaler.fit_transform(df[['Amount']])
    return df





df=pd.read_csv('creditcard.csv')

df=prepoc_data(df)

X_train,X_test=train_test_split(df,test_size=0.3)
X_train=X_train[X_train.Class==0]
X_train=X_train.drop(['Class'],axis=1)
Y_test=X_test['Class']
X_test=X_test.drop(['Class'],axis=1)
X_train = X_train.values
X_test = X_test.values

#Building the model

input_dimension=X_train.shape[1]
encoding_dimension=14

input_l=Input(shape=(input_dimension,))

encoder=Dense(encoding_dimension,activation="tanh",activity_regularizer=regularizers.l1(10e-5))(input_l)

encoder=Dense(encoding_dimension//2,activation="tanh")(encoder)

decoder=Dense(encoding_dimension//2,activation="tanh")(encoder)

decoder = Dense(input_dimension, activation='relu')(decoder)

autoencoder = Model(inputs=input_l, outputs=decoder)

autoencoder.compile(optimizer='adam',
                    loss='mean_squared_error',
                    metrics=['accuracy'])

history = autoencoder.fit(
    X_train, X_train,
    batch_size=64,
    epochs=10,
    validation_data=(X_test, X_test),

)


score = autoencoder.evaluate(X_test, X_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




