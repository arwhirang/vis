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

X = x_train.reshape(60000, 28, 28)
Y = tf.one_hot(y_train, 10)
X = tf.cast(X, tf.float32)
train_tf = tf.data.Dataset.from_tensor_slices((X, Y)).batch(batch_size).shuffle(10000)

print (X.shape, Y.shape)

class IGtest(tf.keras.Model):
    def __init__(self):
        super(IGtest, self).__init__()
        #self.conv1 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')
        #self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        #self.mp1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        #self.dp1 = tf.keras.layers.Dropout(0.25)
        #self.conv3 = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu')
        #self.conv4 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        #self.mp2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))
        #self.dp2 = tf.keras.layers.Dropout(0.25)
        
        #self.dens1 = tf.keras.layers.Dense(128, activation='relu')
        #self.dens2 = tf.keras.layers.Dense(64, activation='relu')
        #self.dp3 = tf.keras.layers.Dropout(0.5)
        self.dens3 = tf.keras.layers.Dense(outChannelInt)

    def call(self, inp, training):
        #lmo = self.conv1(inp)
        #lmo = self.conv2(lmo)
        #lmo = self.mp1(lmo)
        #lmo = self.dp1(lmo, training)
        lmo = tf.keras.layers.Flatten()(inp)
        #lmo = self.dens1(lmo)
        #x = keras.layers.Reshape([196, 1024])(x)
        #lmo = self.dens2(lmo)
        #lmo = self.dp3(lmo, training)
        lmo = self.dens3(lmo)
        pred = tf.nn.softmax(lmo)
        return lmo, pred

meanFunc = tf.keras.metrics.Mean(name='meanFunc')
optimizer = tf.keras.optimizers.Adam(0.001)
lossFunc = tf.keras.losses.CategoricalCrossentropy(from_logits=False, name='lossFunc')
model = IGtest()#tf.keras.Model(inputs=[l_in, l2_in], outputs=x_out)

def loss_function(real, pred_logit):
    #cross_ent = lossFunc(y_true=real, y_pred=pred_logit)
    cross_ent = tf.nn.softmax_cross_entropy_with_logits(logits=pred_logit, labels=real)
    return tf.reduce_sum(cross_ent)

def train_step(inp_X, inp_trainY):
    inp_XV = tf.Variable(inp_X, dtype=tf.float32)
    with tf.GradientTape() as tape:
        #res = model(tf.nn.embedding_lookup(embeddings_, inp_trainX), index, True)
        res, pred = model(inp_XV, True)
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

#https://github.com/ankurtaly/Integrated-Gradients/blob/master/IntegratedGradients/integrated_gradients.py
def ig_scaledInp(inp, baseline, steps=50):
    if baseline is None:
        baseline = tf.zeros_like(inp)
    assert(baseline.shape == inp.shape)


    inp_ = tf.stack([inp for _ in range(steps)])
    baseline_ = tf.stack([baseline for _ in range(steps)])
    # Get difference between sample and reference
    dif = inp_ - baseline_ 
    # Get multipliers
    multiplier = tf.divide(tf.stack([tf.ones_like(inp)*i for i in range(steps)]), steps)
    interploated_dif = tf.multiply(dif, multiplier)
    _shape = [-1] + [int(s) for s in inp.get_shape()[1:]]
    interploated = tf.reshape(interploated_dif, shape=_shape)
    baseline_ = tf.reshape(baseline_, shape=_shape)
    print(interploated.shape, baseline_.shape)
    return interploated, baseline_

    
    
    # Use trapezoidal rule to approximate the integral.
    # See Section 4 of the following paper for an accuracy comparison between
    # left, right, and trapezoidal IG approximations:
    # "Computing Linear Restrictions of Neural Networks", Matthew Sotoudeh, Aditya V. Thakur
    # https://arxiv.org/abs/1908.06214


for tf_x, tf_y in train_tf:
    scaledX, baseline = ig_scaledInp(tf_x, None, steps=3)
    scaledXV = tf.Variable(scaledX, dtype=tf.float32)
    with tf.GradientTape() as g:
        g.watch(scaledXV)
        pred_logit, pred = model(scaledXV, False)
    grads = g.gradient(pred, scaledXV)

    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = np.average(grads, axis = 0)
    integrated_gradients = (scaledXV - baseline)*avg_grads
    print(integrated_gradients.shape)

"""
tf_inter = tf.keras.layers.Concatenate(axis=0)(inter_list) 
tf_preds = tf.keras.layers.Concatenate(axis=0)(preds_list)
tf_steps = tf.keras.layers.Concatenate(axis=0)(step_whole)
"""
 















