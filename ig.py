import numpy as np
import tensorflow as tf
#import matplotlib.pyplot as plt
import TFIntegratedG as ig
#from tensorflow.examples.tutorials.mnist import input_data
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("should be ok...right?")
    except RuntimeError as e:
        print(e)
else:
    print("gpu unlimited?")

print("current pid:", os.getpid())

batch_size = 64
outChannelInt = 10
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

X = np.expand_dims(x_train.reshape(60000, 28, 28), 3)
Y = tf.one_hot(y_train, 10)
X1 = tf.cast(X[:,:,:14], tf.float32)
X2 = tf.cast(X[:,:,14:], tf.float32)
train_tf = tf.data.Dataset.from_tensor_slices((X1, X2, Y)).batch(batch_size).shuffle(10000)

print (X1.shape, Y.shape)

class IGtest(tf.keras.Model):
    def __init__(self):
        super(IGtest, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.mp1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dp1 = tf.keras.layers.Dropout(0.25)

        self.conv3 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')
        self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.mp2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        self.dp2 = tf.keras.layers.Dropout(0.25)
        
        self.dens1 = tf.keras.layers.Dense(128, activation='relu')
        self.dp3 = tf.keras.layers.Dropout(0.5)
        self.dens2 = tf.keras.layers.Dense(outChannelInt, activation='softmax')

        inter, stepsize, ref = ig.linear_inpterpolation(_x, num_steps=15)
        
    def call(self, inp1, inp2, training):
        l = self.conv1(inp1)
        l = self.conv2(l)
        l = self.mp1(l)
        l = self.dp1(l)
        
        l2 = self.conv3(inp2)
        l2 = self.conv4(l2)
        l2 = self.mp2(l2)
        l2 = self.dp2(l2)

        x = tf.keras.layers.Concatenate()([l, l2])
        x = tf.keras.layers.Flatten()(x)
        x = self.dens1(x)
        x = self.dp3(x)
        x_out = self.dens2(x)
        
        return x_out
"""    
l_in = tf.keras.Input(shape=(28, 14, 1), )
l2_in = tf.keras.Input(shape=(28, 14, 1), )
"""
meanFunc = tf.keras.metrics.Mean(name='meanFunc')
optimizer = tf.keras.optimizers.Adam(0.001)
lossFunc = tf.keras.losses.CategoricalCrossentropy(from_logits=False, name='lossFunc')
model = IGtest()#tf.keras.Model(inputs=[l_in, l2_in], outputs=x_out)

def loss_function(real, pred_logit):
    cross_ent = lossFunc(y_true=real, y_pred=pred_logit)
    return tf.reduce_sum(cross_ent)

def train_step(inp_X1, inp_X2, inp_trainY):
    with tf.GradientTape() as tape:
        #res = model(tf.nn.embedding_lookup(embeddings_, inp_trainX), index, True)
        res = model(inp_X1, inp_X2, True)
        loss = loss_function(inp_trainY, res)
    
    print(model.variables)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    meanFunc.update_state(loss)
    
   
#bestAUC = 0
#bestEpoch = 0
#model.save_weights(checkpoint_dir)
for epoch in range(15):
    meanFunc.reset_states()
    #start = time.time()
    for tf_x1, tf_x2, tf_y1 in train_tf:
        train_step(tf_x1, tf_x2, tf_y1)
            
#model.fit([X1, X2], Y, epochs=15, batch_size=128, verbose=0)
#predicted = model.predict([X1, X2])
ig = integrated_gradients(model, [X1, X2], optimizer, outChannelInt) # inputs=[X1, X2], opt=optimizer, outChannelInt)


index = np.random.randint(55000)
pred = np.argmax(predicted[index])
print ("prediction:", pred)

################# Calling Explain() function #############################
ex = ig.explain([X1[index, :, :, :], X2[index, :, :, :]], outc=pred)
##########################################################################

th = max(np.abs(np.min([np.min(ex[0]), np.min(ex[1])])), np.abs(np.max([np.max(ex[0]), np.max(ex[1])])))
