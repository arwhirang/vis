import pickle
import numpy as np

import tensorflow as tf
import os
from VisUtil import *

###############################
#set the gpu parameters for tensorflow 2
###############################
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#", 1, 2, 3"
print("current pid:", os.getpid())

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

USE_DUMPED_DATA = True#After the significant SA dump is made, just load the dump for quick process
TOX21_FILE = "NR-AR"
SARPY_FILE = "nr_ar"
BIOALERT_FILE = "nr-ar_bioalerts"
NN_FILE_NUMBER = "2"
ZSCORE = 2.58


#load validation and test for comparison.
def loadValTest(NN_FILE_NUMBER):
    file_test = "saved/save_igtest" + NN_FILE_NUMBER
    file_valid = "saved/save_igvalid" + NN_FILE_NUMBER
    IGweights2, teststr2 = pickle.load(open(file_test, "rb"))
    IGweights3, teststr3 = pickle.load(open(file_valid, "rb"))
    return IGweights2, teststr2, IGweights3, teststr3


#make a dict of SMILES from Val and Test set for comparison.
def dicSMILESValTest(teststr2, teststr3):
    dictValAndTest = {}
    valAndTest = teststr2 + teststr3
    for currSmiList in valAndTest:
        currsmi = "".join(currSmiList)
        dictValAndTest[currsmi] = 1
    return dictValAndTest
    
    
# make subst_mols_NN dict for ease of code
def GenSubstMolsNN(extractedSigSADict):
    subst_mols_NN = {}  # key, val = <substructure mol, list of (original mol, list of amap)>
    for key, val in extractedSigSADict.items():
        submol = Chem.MolFromSmarts(key)
        subst_mols_NN[submol] = val
    return subst_mols_NN
    
    
#############################
# main code starts
if not USE_DUMPED_DATA:#After the significant SA dump is made, just load the dump for quick process
    file_train = "saved/save_igtrain" + NN_FILE_NUMBER
    IGweights1, teststr1 = pickle.load(open(file_train, "rb"))
    
    # make dicSAs for all SAs
    # key, value: <substr in str, list of scores> / each of the scores are the sum of the atoms in the smart
    dicSAs = {}
    # make dicOriAmap to store original Mol and SAs
    # key, value: <substr in str, [orimol, list of amap]>
    dicOriAmap = {}
    layer1_sum = tf.math.reduce_max(IGweights1, axis=2)  # layer1_max shape = (whole_size, seq_len)
    layer_sum = layer1_sum#tf.keras.layers.concatenate([layer1_sum, layer2_sum, layer3_sum], axis=0)
    teststrs = teststr1# + teststr2 + teststr3
    assert len(layer_sum) == len(teststrs)

    shared, notShared = 0, 0
    for i, currSmiList in enumerate(teststrs):
        currsmi = "".join(currSmiList)
        currmol = Chem.MolFromSmiles(currsmi)

        th_Until7thW = sorted(layer_sum[i])[-7]
        #pick the significant indicies that have high enough weights above certain threshold (7th elements from the biggest)
        sigIdxList = idxForSignificants(th_Until7thW, layer_sum[i])
        #Since atoms are our main concern, separate the indicies of atoms from the significant indicies 
        sigIdx_general, sigIdx_atom = sigIdxAtom(sigIdxList, currSmiList)
        #creates submolecules from the significant atoms using rdkit
        #also, create orimol_atomI tuple. orimol_atomI is for the comparison part
        dicSubstrAmap, orimol_atomI = get_substruct(currmol, sigIdx_atom)
        # the atommap of the rdkit calculates atom indicies while integrated gradients are based on SMILES.
        # this func changes the atommap indicies into a SMILES indicies 
        dicReal = get_real_indicies(dicSubstrAmap, currSmiList, bond_score_include=False)
        # get_scores function updates dicSAs and dicOriMap
        get_scores(dicReal, layer_sum[i], orimol_atomI, dicSAs, dicOriAmap)

    # calculate global mean score of every substrs
    gmSMA = {}
    for key, val in dicSAs.items():
        gmSMA[key] = np.mean(val)
        
    # make the score list into z-score arrays
    keylist, zscorearray = getZscores(gmSMA)
    # extract importnat structures that contain high z-scores
    # key, val = <substr in str, list of (orimol, list of amap)>
    extractedSigSADict = extSigSA(keylist, zscorearray, dicOriAmap, ZSCORE)
    pickle.dump(extractedSigSADict, open("extractedSA"+ NN_FILE_NUMBER +".pickle", "wb"))
else:
    extractedSigSADict = pickle.load(open("extractedSA"+ NN_FILE_NUMBER +".pickle", "rb"))
    
print("num of extracted SAs: ", len(extractedSigSADict))
subst_mols_NN = GenSubstMolsNN(extractedSigSADict)


#validation and test for comparison.
IGweights2, teststr2, IGweights3, teststr3 = loadValTest(NN_FILE_NUMBER)
dictValAndTest = dicSMILESValTest(teststr2, teststr3)#make a dict of SMILES from Val and Test set
    


#############################
# comparison code starts
# load SAs from bioalerts
bioalertFname = "refData/" + BIOALERT_FILE
bioalert_subst = pickle.load(open(bioalertFname, "rb"))#actually the data types are different from bioalerts and sarpy
subst_mols_bioalert = load_bioalert(bioalert_subst)
subst_mols_bioalertU = rdkit_unique(subst_mols_bioalert)

# load SAs from sarpy
subst_mols_sarpy = load_sarpy(SARPY_FILE)

#comparison code
compareWithAtomUnits(subst_mols_NN, subst_mols_bioalertU)
compareWithAtomUnits(subst_mols_NN, subst_mols_sarpy)







###############################
####Not in the poster paper####
###############################

# For better analysis, we will calculate ratio for positive that contains SAs
def loadTox21Field(proteinName):
    listX, listY = [], []
    listXsmi = []
    afile = './TOX21/' + proteinName + '_wholetraining.smiles'
    f = open(afile, "r")
    lines = f.readlines()
    for line in lines:
        splitted = line.split(" ")
        if len(splitted[0]) >= 200:
            continue
        listX.append(Chem.MolFromSmiles(splitted[0]))
        listXsmi.append(splitted[0])
        listY.append(float(splitted[1]))
    f.close()
    return listX, listY, listXsmi


listX, listY, listXsmi = loadTox21Field(TOX21_FILE)


def computeONLYStat(listX, listY, currModel, currModelName):
    cntOnlyP, cntOnlyN = 0, 0
    for submol in currModel.keys():
        prevLable = 0
        isMixed = False
        for idx, orimol in enumerate(listX):
            if orimol.HasSubstructMatch(submol):
                if listY[idx] == 1:
                    if prevLable == -1:
                        isMixed = True
                    prevLable = 1
                else:
                    if prevLable == 1:
                        isMixed = True
                    prevLable = -1
        if isMixed == False:
            if prevLable == 1:
                cntOnlyP += 1
            elif prevLable == -1:
                cntOnlyN += 1

    print("SAs from", currModelName, cntOnlyP, cntOnlyN)

print("            cntOnlyP, cntOnlyN")
computeONLYStat(listX, listY, subst_mols_NN, "NN")
computeONLYStat(listX, listY, subst_mols_bioalertU, "bioialerts")
computeONLYStat(listX, listY, subst_mols_sarpy, "sarpy")


#the difference between above code: screening only in the validation and the test set
def computeValTestStat(listX, listY, listXsmi, currModel, currModelName, dictValAndTest):
    cntOnlyP, cntOnlyN = 0, 0
    for submol in currModel.keys():
        prevLable = 0
        isMixed = False
        for idx, orimol in enumerate(listX):
            if listXsmi[idx] not in dictValAndTest:
                continue
            if orimol.HasSubstructMatch(submol):
                if listY[idx] == 1:
                    if prevLable == -1:
                        isMixed = True
                    prevLable = 1
                else:
                    if prevLable == 1:
                        isMixed = True
                    prevLable = -1
        if isMixed == False:
            if prevLable == 1:
                cntOnlyP += 1
            elif prevLable == -1:
                cntOnlyN += 1

    print("ValTestStat  SAs from", currModelName, cntOnlyP, cntOnlyN)
computeValTestStat(listX, listY, listXsmi, subst_mols_NN, "NN", dictValAndTest)
computeValTestStat(listX, listY, listXsmi, subst_mols_bioalertU, "bioialerts", dictValAndTest)
computeValTestStat(listX, listY, listXsmi, subst_mols_sarpy, "sarpy", dictValAndTest)
