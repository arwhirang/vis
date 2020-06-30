import os, sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, recall_score
#import matplotlib.pyplot as plt
import time
from rdkit.Chem import AllChem
from rdkit import Chem
from sklearn.utils import shuffle
import pickle
import itertools

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

# hyperparameters
batch_size = 51
Max_len = 200 # for padding
embedding_size = 200
n_hid = 1024 # number of feature maps
win_size = 5 # window size of kernel
lr = 1e-4 # learning rate of optimzier

# lookup table
bit_size = 1024 # circular fingerprint
emb = tf.Variable(tf.random.uniform([bit_size, embedding_size], -1, 1), dtype=tf.float32)
pads = tf.constant([[1,0], [0,0]])
embeddings_ = tf.pad(emb, pads)#1025, 200
pickle_load = False

# load data =========================================
def posNegNums(ydata):
    cntP = 0
    cntN = 0
    for ele in ydata:
        if ele == 1:
            cntP += 1
        else:
            cntN += 1
    return cntP, cntN


def char2indices(listStr, dicC2I):
    listIndices = [0] * 200
    charlist = listStr
    size = len(listStr)
    twoChars = {"AL":1, "al":1, "Al":1, "AU":1, "au":1, "Au":1, "MG":1, "mg":1, "Mg":1, "ZN":1, "zn":1, "Zn":1, "CA":1, "ca":1, "Ca":1, "NA":1, "na":1, "Na":1, "CL":1, "cl":1, "Cl":1, "FE":1, "fe":1, "Fe":1, "BR":1, "br":1, "Br":1, "SI":1, "si":1, "Si":1, "BI":1, "Bi":1, "bi":1, "GE":1, "Ge":1, "ge":1, "CU":1, "Cu":1, "cu":1, "SN":1, "Sn":1, "sn":1, "TL":1, "Tl":1, "tl":1, "PT":1, "Pt":1, "pt":1, "PD":1, "Pd":1, "pd":1, "AS":1, "As":1, "as":1, "CR":1, "Cr":1, "cr":1, "CD":1, "Cd":1, "cd":1, "BE":1, "Be":1, "be":1, "SR":1, "Sr":1, "sr":1, "ZR":1, "Zr":1, "zr":1, "BA":1, "Ba":1, "ba":1, "MO":1, "Mo":1, "mo":1, "TI":1, "Ti":1, "ti":1, "SB":1, "Sb":1, "sb":1, "NI":1, "Ni":1, "ni":1, "ND":1, "Nd":1, "nd":1, "IN":1, "In":1, "in":1, "SE":1, "Se":1, "se":1, "PB":1, "Pb":1, "pb":1}
    prevTwoCharsFlag = False
    indexForList = 0
    for i, c in enumerate(charlist):
        if prevTwoCharsFlag:
            prevTwoCharsFlag = False
            continue
        
        if i != size - 1 and "".join(charlist[i:i+2]) in twoChars:
            two = "".join(charlist[i:i+2])
            if two not in dicC2I:
                dicC2I[two] = len(dicC2I) + 1
                listIndices[indexForList] = dicC2I[two]
                indexForList += 1
            else:
                listIndices[indexForList] = dicC2I[two]
                indexForList += 1
            prevTwoCharsFlag = True
        else:    
            if c not in dicC2I:
                dicC2I[c] = len(dicC2I) + 1
                listIndices[indexForList] = dicC2I[c]
                indexForList += 1
            else:
                listIndices[indexForList] = dicC2I[c]
                indexForList += 1
    return listIndices


def makeDataForSmilesOnly(proteinName, dicC2I):
    listX, listY = [], []
    afile = 'TOX21/' + proteinName + '_wholetraining.smiles'
    f = open(afile, "r")
    lines = f.readlines()
    for line in lines:
        splitted = line.split(" ")
        if len(splitted[0]) >= 200:
            continue
        listX.append(char2indices(splitted[0], dicC2I))  # length can vary
        listY.append(float(splitted[1]))
    f.close()
    # print("how many weird cases exist?", cntTooLong, weirdButUseful)
    train_x, test_x, train_y, test_y = train_test_split(listX, listY, test_size=0.1)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1111)
    pos_num, neg_num = posNegNums(train_y)
    train_tf = tf.data.Dataset.from_tensor_slices((train_x, train_y)).batch(batch_size).shuffle(10000)
    valid_tf = tf.data.Dataset.from_tensor_slices((valid_x, valid_y)).batch(batch_size).shuffle(10000)
    test_tf = tf.data.Dataset.from_tensor_slices((test_x, test_y)).batch(batch_size)
    return train_tf, valid_tf, test_tf, pos_num, neg_num, test_x


if pickle_load:
    embeddings_, dicC2I = pickle.load(open("saved_emb.pkl", "rb"))
else:
    dicC2I = {}
pos_num, neg_num = 0, 0
train_tf1, valid_tf1, test_tf1, pos1, neg1, test1 = makeDataForSmilesOnly("NR-AR-LBD", dicC2I)
train_tf2, valid_tf2, test_tf2, pos2, neg2, test2 = makeDataForSmilesOnly("NR-AR", dicC2I)
train_tf3, valid_tf3, test_tf3, pos3, neg3, test3 = makeDataForSmilesOnly("NR-AhR", dicC2I)
train_tf4, valid_tf4, test_tf4, pos4, neg4, test4 = makeDataForSmilesOnly("NR-Aromatase", dicC2I)
train_tf5, valid_tf5, test_tf5, pos5, neg5, test5 = makeDataForSmilesOnly("NR-ER-LBD", dicC2I)
train_tf6, valid_tf6, test_tf6, pos6, neg6, test6 = makeDataForSmilesOnly("NR-ER", dicC2I)
train_tf7, valid_tf7, test_tf7, pos7, neg7, test7 = makeDataForSmilesOnly("NR-PPAR-gamma", dicC2I)
train_tf8, valid_tf8, test_tf8, pos8, neg8, test8 = makeDataForSmilesOnly("SR-ARE", dicC2I)
train_tf9, valid_tf9, test_tf9, pos9, neg9, test9 = makeDataForSmilesOnly("SR-ATAD5", dicC2I)
train_tf10, valid_tf10, test_tf10, pos10, neg10, test10 = makeDataForSmilesOnly("SR-HSE", dicC2I)
train_tf11, valid_tf11, test_tf11, pos11, neg11, test11 = makeDataForSmilesOnly("SR-MMP", dicC2I)
train_tf12, valid_tf12, test_tf12, pos12, neg12, test12 = makeDataForSmilesOnly("SR-p53", dicC2I)
pos_num = pos1 + pos2 + pos3 + pos4 + pos5 + pos6 + pos7 + pos8 + pos9 + pos10 + pos11 + pos12
neg_num = neg1 + neg2 + neg3 + neg4 + neg5 + neg6 + neg7 + neg8 + neg9 + neg10 + neg11 + neg12
print("pos/neg:", pos_num, neg_num)

##################################################################
# ================== CNN model construction ======================
##################################################################
class CustomHot(keras.layers.Layer):  
    def __init__(self):
        super(CustomHot, self).__init__()

    def call(self, inputs):
        return tf.one_hot(inputs, 12)

class CustomRSum(keras.layers.Layer):  
    def __init__(self):
        super(CustomRSum, self).__init__()

    def call(self, inputs, dWhich):
        return tf.math.reduce_sum(inputs*dWhich, axis=1)

class fp2vec(tf.keras.Model):
    def __init__(self, output_bias):
        super(fp2vec, self).__init__()
        if output_bias is not None:
            output_bias = tf.keras.initializers.Constant(output_bias)
        self.conv1 = keras.layers.Conv2D(1024, [5, 200], strides=1)
        self.pads1 = tf.constant([[0, 0], [0, 5-1], [0, 0]])
        self.maxp = keras.layers.MaxPool1D([1024], data_format='channels_first')
        
        self.dens1 = keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
        self.dens2 = keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
        self.dens3 = keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
        self.dens4 = keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
        self.dens5 = keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
        self.dens6 = keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
        self.dens7 = keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
        self.dens8 = keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
        self.dens9 = keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
        self.dens10 = keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
        self.dens11 = keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
        self.dens12 = keras.layers.Dense(1, activation='sigmoid', bias_initializer=output_bias)
        
    def call(self, inp, whichClass, training):
        x = keras.layers.Reshape([200, 200, 1])(inp)
        x = self.conv1(x)
        x = keras.layers.Reshape([196, 1024])(x)
        x = tf.pad(x, self.pads1)# (batch_size, seq_len, self.d_model)
        x_ = self.maxp(x)
        x_ = keras.layers.Dropout(0.5)(x_, training)
        x_ = keras.layers.Reshape([200])(x_)
        cl1 = self.dens1(x_)
        cl2 = self.dens2(x_)
        cl3 = self.dens3(x_)
        cl4 = self.dens4(x_)
        cl5 = self.dens5(x_)
        cl6 = self.dens6(x_)
        cl7 = self.dens7(x_)
        cl8 = self.dens8(x_)
        cl9 = self.dens9(x_)
        cl10 = self.dens10(x_)
        cl11 = self.dens11(x_)
        cl12 = self.dens12(x_)
        x_out = keras.layers.concatenate([cl1, cl2, cl3, cl4, cl5, cl6, cl7, cl8, cl9, cl10, cl11, cl12])
        decideWhich = CustomHot()(whichClass)
        outputClass = CustomRSum()(x_out, decideWhich)
        return outputClass

#train the model
optimizer = keras.optimizers.Adam(lr)

meanFunc = keras.metrics.Mean(name='meanFunc')
accFunc = keras.metrics.BinaryAccuracy()
aucFunc = keras.metrics.AUC()
precFunc = keras.metrics.Precision()
recallFunc = keras.metrics.Recall()
lossFunc = keras.losses.BinaryCrossentropy(from_logits=False, name='lossFunc')
initial_bias = np.log([pos_num / neg_num])
#weight_for_0 = tf.convert_to_tensor((1 / neg_num)*(pos_num + neg_num)/2.0, dtype=tf.float32)#not used 
weight_for_1 = tf.convert_to_tensor((1 / pos_num)*(pos_num + neg_num)/2.0, dtype=tf.float32)
model = fp2vec(initial_bias)

checkpoint_dir = "tr1/cp.ckpt"

def loss_function(real, pred_, sampleW):
    cross_ent = lossFunc(y_true=real, y_pred=pred_)
    #cross_ent = tf.nn.weighted_cross_entropy_with_logits(logits=pred_logit, labels=real, pos_weight=sampleW)
    return tf.reduce_sum(cross_ent)

def train_step(inp_trainX, inp_trainY, whichClass):
    with tf.GradientTape() as tape:
        res = model(tf.nn.embedding_lookup(embeddings_, inp_trainX), whichClass, True)
        #print(predictions, inp_trainY)
        loss = loss_function(inp_trainY, res, sampleW=weight_for_1)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    meanFunc.update_state(loss)
    
    
def eval_step(inp_valX, reals, whichClass):
    preds = model(tf.nn.embedding_lookup(embeddings_, inp_valX), whichClass, False)
    #y_pred_ = preds
    
    precFunc.update_state(y_true=reals, y_pred=preds)
    recallFunc.update_state(y_true=reals, y_pred=preds)
    aucFunc.update_state(y_true=reals, y_pred=preds)
    accFunc.update_state(y_true=reals, y_pred=preds)


# https://github.com/ankurtaly/Integrated-Gradients/blob/master/IntegratedGradients/integrated_gradients.py
def ig_scaledInp(inp, baseline, steps=50):
    if baseline is None:
        baseline = tf.zeros_like(inp)
    assert (baseline.shape == inp.shape)
    scaled_inputs = baseline
    for i in range(0, steps + 1):
        scaled_inputs = scaled_inputs + (float(i)/steps)*(inp - baseline)
    return scaled_inputs, baseline
    
    
def IGtest_step(inp_, real, whichClass):
    all_intgrads = []
    embedded_inp = tf.nn.embedding_lookup(embeddings_, inp_)
    for i in range(5):
        randBaseFor1 = np.random.random_sample((200, embedding_size))
        randBase = [randBaseFor1 for i in range(inp_.shape[0])]
        tfRandBase = tf.Variable(randBase, dtype=tf.float32)
        scaledX, baseline = ig_scaledInp(embedded_inp, tfRandBase, steps=3)
        scaledXV = tf.Variable(scaledX, dtype=tf.float32)
        with tf.GradientTape() as g:
            g.watch(scaledXV)
            pred = model(scaledXV, whichClass, False)

        grads = g.gradient(pred, scaledXV)
        #print("grads.shape:",grads.shape)
        grads = (grads[:-1] + grads[1:]) / 2.0
        avg_grads = tf.math.reduce_mean(grads, axis=0, keepdims=True)
        intgrads = (scaledXV - baseline) * avg_grads
        #print("intgrads.shape:",intgrads.shape)
        all_intgrads.append(intgrads)
    avg_intgrads = tf.math.reduce_mean(all_intgrads, axis=0)
    return avg_intgrads



bestAUC = 0
bestEpoch = 0
model.save_weights(checkpoint_dir)
for epoch in range(71):
    meanFunc.reset_states()
    precFunc.reset_states()
    recallFunc.reset_states()
    aucFunc.reset_states()
    accFunc.reset_states()
    start = time.time()
    for tf1, tf2, tf3, tf4, tf5, tf6, tf7, tf8, tf9, tf10, tf11, tf12 in itertools.zip_longest(train_tf1, train_tf2, train_tf3, train_tf4, train_tf5, train_tf6, train_tf7, train_tf8, train_tf9, train_tf10, train_tf11, train_tf12):
        if tf1:
            train_step(tf1[0], tf1[1], 0)
        if tf2:
            train_step(tf2[0], tf2[1], 1)
        if tf3:
            train_step(tf3[0], tf3[1], 2)
        if tf4:
            train_step(tf4[0], tf4[1], 3)
        if tf5:
            train_step(tf5[0], tf5[1], 4)
        if tf6:
            train_step(tf6[0], tf6[1], 5)
        if tf7:
            train_step(tf7[0], tf7[1], 6)
        if tf8:
            train_step(tf8[0], tf8[1], 7)
        if tf9:
            train_step(tf9[0], tf9[1], 8)
        if tf10:
            train_step(tf10[0], tf10[1], 9)
        if tf11:
            train_step(tf11[0], tf11[1], 10)
        if tf12:
            train_step(tf12[0], tf12[1], 11)

    if epoch%2 == 0:
        for tf1, tf2, tf3, tf4, tf5, tf6, tf7, tf8, tf9, tf10, tf11, tf12 in itertools.zip_longest(valid_tf1, valid_tf2, valid_tf3, valid_tf4, valid_tf5, valid_tf6,
                                             valid_tf7, valid_tf8, valid_tf9, valid_tf10, valid_tf11, valid_tf12):
            if tf1:
                eval_step(tf1[0], tf1[1], 0)
            if tf2:
                eval_step(tf2[0], tf2[1], 1)
            if tf3:
                eval_step(tf3[0], tf3[1], 2)
            if tf4:
                eval_step(tf4[0], tf4[1], 3)
            if tf5:
                eval_step(tf5[0], tf5[1], 4)
            if tf6:
                eval_step(tf6[0], tf6[1], 5)
            if tf7:
                eval_step(tf7[0], tf7[1], 6)
            if tf8:
                eval_step(tf8[0], tf8[1], 7)
            if tf9:
                eval_step(tf9[0], tf9[1], 8)
            if tf10:
                eval_step(tf10[0], tf10[1], 9)
            if tf11:
                eval_step(tf11[0], tf11[1], 10)
            if tf12:
                eval_step(tf12[0], tf12[1], 11)
        print('Valid: prec {:.4f} recall {:.4f} AUC {:.4f}, acc {:.4f}'.format( precFunc.result(), recallFunc.result(), aucFunc.result(), accFunc.result()))
        if bestAUC < aucFunc.result():
            bestEpoch = epoch + 1
            bestAUC = aucFunc.result()
            model.save_weights(checkpoint_dir)
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, checkpoint_dir))
    print('epoch {} train Loss {:.4f}'.format(epoch + 1, meanFunc.result()))
    print('Time taken for current epoch: {} secs\n'.format(time.time() - start))        


precFunc.reset_states()
recallFunc.reset_states()
aucFunc.reset_states()
accFunc.reset_states()

model.load_weights(checkpoint_dir)
print("weights loaded from the epoch:", bestEpoch)
for X, Y in test_tf1:
    eval_step(X, Y, 0)
for X, Y in test_tf2:
    eval_step(X, Y, 1)
for X, Y in test_tf3:
    eval_step(X, Y, 2)
for X, Y in test_tf4:
    eval_step(X, Y, 3)
for X, Y in test_tf5:
    eval_step(X, Y, 4)
for X, Y in test_tf6:
    eval_step(X, Y, 5)
for X, Y in test_tf7:
    eval_step(X, Y, 6)
for X, Y in test_tf8:
    eval_step(X, Y, 7)
for X, Y in test_tf9:
    eval_step(X, Y, 8)
for X, Y in test_tf10:
    eval_step(X, Y, 9)
for X, Y in test_tf11:  
    eval_step(X, Y, 10)
for X, Y in test_tf12:
    eval_step(X, Y, 11)
print('Test : prec {:.4f} recall {:.4f} AUC {:.4f}, acc {:.4f}'.format( precFunc.result(), recallFunc.result(), aucFunc.result(), accFunc.result()))


print("start analyzing")

def indices2chars(listx, dicC2I):#listx shape = whole_size, 200
    dicI2C = {}
    for key in dicC2I:
        dicI2C[dicC2I[key]] = key

    retList = []
    for instance in listx:
        tmplist = []
        for index in instance:
            if index == 0:#didn't use the 0 as index
                break
            tmplist.append(dicI2C[index])
        retList.append(tmplist)
    return retList
    
#for the NR-AR domain...
ig_res = []
for tf2 in test_tf2:
    ig_res.extend(IGtest_step(tf2[0], tf2[1], 1))
    
pickle.dump((ig_res, indices2chars(test2, dicC2I)), open( "save_igtest2", "wb" ) )

ig_res = []
for tf2 in valid_tf2:
    ig_res.extend(IGtest_step(tf2[0], tf2[1], 1))

pickle.dump((ig_res, indices2chars(valid2, dicC2I)), open( "save_igvalid2", "wb" ) )

ig_res = []
for tf2 in train_tf2:
    ig_res.extend(IGtest_step(tf2[0], tf2[1], 1))

pickle.dump((ig_res, indices2chars(train2, dicC2I)), open( "save_igtrain2", "wb" ) )

