import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
from IntegratedGradients import *
#from tensorflow.examples.tutorials.mnist import input_data

#mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

X = np.expand_dims(x_train.reshape(60000, 28, 28), 3)
Y = tf.one_hot(y_train, 10)
X1 = X[:,:,:14]
X2 = X[:,:,14:]

print (X1.shape, Y.shape)

l_in = tf.keras.Input(shape=(28, 14, 1), )
l = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(l_in)
l = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(l)
l = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(l)
l = tf.keras.layers.Dropout(0.25)(l)

l2_in = tf.keras.Input(shape=(28, 14, 1), )
l2 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu') (l2_in)
l2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(l2)
l2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(l2)
l2 = tf.keras.layers.Dropout(0.25)(l2)

x = tf.keras.layers.concatenate([l, l2])

x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)
x_out = tf.keras.layers.Dense(10, activation='softmax')(x)

model = tf.keras.Model(inputs=[l_in, l2_in], outputs=x_out)
model.compile(optimizer='sgd', loss='categorical_crossentropy')
model.fit([X1, X2], Y, epochs=15, batch_size=128, verbose=0)
predicted = model.predict([X1, X2])
ig = integrated_gradients(model)


index = np.random.randint(55000)
pred = np.argmax(predicted[index])
print ("prediction:", pred)

################# Calling Explain() function #############################
ex = ig.explain([X1[index, :, :, :], X2[index, :, :, :]], outc=pred)
##########################################################################

th = max(np.abs(np.min([np.min(ex[0]), np.min(ex[1])])), np.abs(np.max([np.max(ex[0]), np.max(ex[1])])))
