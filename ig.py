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
X = tf.cast(X, tf.float32)
train_tf = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size).shuffle(10000)

print (X.shape, Y.shape)

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

    def call(self, inp, training):
        l = self.conv1(inp)
        l = self.conv2(l)
        l = self.mp1(l)
        l = self.dp1(l)
        l = tf.keras.layers.Flatten()(l)
        l = self.dens1(l)
        l = self.dp3(l)
        x_out = self.dens2(l)

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

def train_step(inp_X, inp_trainY):
    with tf.GradientTape() as tape:
        #res = model(tf.nn.embedding_lookup(embeddings_, inp_trainX), index, True)
        res = model(inp_X, True)
        loss = loss_function(inp_trainY, res)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    meanFunc.update_state(loss)
    
   
#bestAUC = 0
#bestEpoch = 0
#model.save_weights(checkpoint_dir)
for epoch in range(15):
    meanFunc.reset_states()
    #start = time.time()
    for tf_x, tf_y in train_tf:
        train_step(tf_x, tf_y)

inter_list = []
preds_list = []
step_whole = []
for tf_x, tf_y in train_tf:
    inter, stepsize, ref = ig.linear_inpterpolation(tf_x, num_steps=15)
    inter_list.extend(inter)
    step_whole.extend(stepsize)
    tmpPred = model(inter, False)
    preds_list.append(tmpPred)

explanations = []
for i in range(10):
    explanations.append(ig.build_ig(inter_list, step_whole, preds_list[:, i], num_steps=15))
        
 















