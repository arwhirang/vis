import pickle
import os
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from itertools import product
from PIL import Image, ImageOps

from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, AveragePooling1D
from tensorflow.keras.layers import Dense, Dropout, ZeroPadding1D, Dot, Reshape, Concatenate
from tensorflow.keras import optimizers
from sklearn.metrics import roc_auc_score
"""
import tensorflow as tf
from tensorflow.keras import optimizers
from sklearn.model_selection import train_test_split
from AMES_aux import *

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


#parameters for data generation
pad_len = 1200 #maximal length of data
asym = ['Li', 'Na', 'K', 'Be', 'Mg', 'Ca', 'Sc', 'Cr', 'Fe', 'Pt', 'Cu', 'Hg', 'B', 'Al', 'C0', 'C1', 'C2', 'C3', 
        'Si', 'Sn', 'N0', 'N1', 'N2', 'P0', 'As', 'Sb', 'O0', 'O1', 'S0', 'S1', 'Se', 'F', 'Cl', 'Br', 'I']
arom = ['c', 'n', 'o', 's']
other = ['X']
one_hot = asym + arom + other
#model parameters
nr_atoms = int(pad_len/8)
input_shape = (nr_atoms,len(one_hot)+4)
output_shape = 1
filters = [1024, 1024, 1024]
dense_layers = [512]
n = 3
batch_size = 64
opt = optimizers.Adam(lr=0.001)

def makeDataForSmilesOnly(proteinName):
    listX, listY = [], []
    afile = 'TOX21/' + proteinName + '_wholetraining.smiles'
    f = open(afile, "r")
    lines = f.readlines()
    for line in lines:
        splitted = line.split(" ")
        if len(splitted[0]) >= 200:
            continue
        listX.append(str(splitted[0]))
        listY.append(float(splitted[1]))
    f.close()        
    train_x, test_x, train_y, test_y = train_test_split(listX, listY, test_size=0.1)
    train_x, valid_x, train_y, valid_y = train_test_split(train_x, train_y, test_size=0.1111)
    return train_x, train_y, valid_x, valid_y, test_x, test_y

train_x, train_y, valid_x, valid_y, test_x, test_y = makeDataForSmilesOnly("NR-AR")

#data generation from scratch
xVal, pVal = Generator(valid_x, one_hot, pad_len)
xTest, pTest = Generator(test_x, one_hot, pad_len)
xTrain, pTrain = Generator(train_x, one_hot, pad_len)

model = create_model(input_shape, output_shape, filters, dense_layers, opt)

model, tr, val, tst = run_model(model, [xTrain, pTrain], train_y, [xVal, pVal], valid_y, [xTest, pTest], test_y, 
                                n=n, batch_size=batch_size, model_file='ames_from_scratch')

pval = model.predict([xVal, pVal])
ptst = model.predict([xTest, pTest])

roc_auc_score(valid_y, pval), roc_auc_score(test_y, ptst)

x = xVal
p = pVal
y = valid_y
mols = valid_x
#layers are a list of indices of the atom representation layers
#returns an array of atom representations and molecule representations
atom_act, mol_act = get_activations(model, x, p, layers=[5,9,13,15], filters=filters, bs=8, pad_len=pad_len)
unique, idx, a_idx = reduce_act(atom_act, mols, nr_atoms=nr_atoms)
