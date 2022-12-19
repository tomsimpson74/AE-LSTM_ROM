import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import os
from natsort import natsorted
from glob import glob
import matplotlib.pyplot as plt

X_T = np.load('X_Train.npy')
X_train = np.zeros((X_T.shape[0]*X_T.shape[2],X_T.shape[1]))
for i in range(X_T.shape[2]):
    X_train[i*X_T.shape[0]:(i+1)*X_T.shape[0],:]=X_T[:,:,i]

X_T2 = np.load('X_Test.npy')
X_test = np.zeros((X_T2.shape[0]*X_T2.shape[2],X_T2.shape[1]))
print(X_T2.shape)
for i in range(X_T2.shape[2]):
    X_test[i*X_T2.shape[0]:(i+1)*X_T2.shape[0],:]=X_T2[:,:,i]

latent_dim = 4
usualCallback = EarlyStopping(monitor='val_loss', min_delta=1e-9, patience = 50)
mcp_save = ModelCheckpoint('model_wts.hdf5', save_best_only=True, monitor='val_loss', mode='min')
inputVector = tf.keras.Input(shape=(X_train.shape[1],))
encoded = layers.Dense(128,activation='tanh')(inputVector)
encoded = layers.Dense(64, activation='tanh')(encoded)
encoded = layers.Dense(latent_dim, activation='linear')(encoded)
decoded = layers.Dense(64, activation='tanh')(encoded)
decoded = layers.Dense(128, activation='tanh')(decoded)
decoded = layers.Dense(X_train.shape[1], activation='linear')(decoded)

autoencoder = tf.keras.Model(inputVector, decoded)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
autoencoder.compile(optimizer=optimizer, loss='mean_squared_error')

encoder = tf.keras.Model(inputVector,encoded)
decoder = tf.keras.Model(encoded,decoded)


'''
autoencoder.fit(X_train, X_train,
                     epochs=50,
                     batch_size=512,
                     shuffle=True,
                     validation_data=(X_test[:30000,:], X_test[:30000,:]),
                     callbacks=[usualCallback,mcp_save])
'''
autoencoder.load_weights('model_wts.hdf5')

weightsList = autoencoder.get_weights()
encoder.set_weights(weightsList[:6])
decoder.set_weights(weightsList[6:])
decoder.save('decoder')


Z_train = encoder.predict(X_train)
Z_test = encoder.predict(X_test)

np.save('Z_train.npy',Z_train)
np.save('Z_test.npy',Z_test)
