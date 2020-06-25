import pickle
import numpy as np
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
from IPython.display import SVG
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem import Draw
import tensorflow as tf
import os
from scipy import stats

os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
print("current pid:", os.getpid())

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_visible_devices(gpus[2], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[2], True)
        print("should be ok...right?")
    except RuntimeError as e:
        print(e)
else:
    print("gpu unlimited?")

USE_DUMPED_DATA = True


def idxForSignificants(th_, aWeight):
    retlist = []
    for idx, ele in enumerate(aWeight):
        if ele >= th_:
            retlist.append(idx)
    return retlist

def sigIdxAtom(sigIdx, currSmiList):
    sigIdx_general, sigIdx_atom = [], []
    twoChars = {"Al": 1, "Au": 1, "Ag": 1, "As": 1, "Ba": 1, "Be": 1, "Bi": 1, "Br": 1, "Ca": 1, "Cd": 1, "Cl": 1,
                "Co": 1, "Cr": 1, "Cu": 1, "Dy": 1, "Fe": 1, "Gd": 1, "Ge": 1, "In": 1, "Li": 1, "Mg": 1, "Mn": 1,
                "Mo": 1, "Na": 1, "Ni": 1, "Nd": 1, "Pb": 1, "Pt": 1, "Pd": 1, "Ru": 1, "Sb": 1, "Se": 1, "se": 1,
                "Si": 1, "Sn": 1, "Sr": 1, "Ti": 1, "Tl": 1, "Yb": 1, "Zn": 1, "Zr": 1}
    atomcnt = -1#first atom is 0
    for idx, ele in enumerate(currSmiList):
        if ele in twoChars or ele.isalpha():# or ele.isdigit():
            if ele == "@" or ele == "H":#these are the same as the bond signals
                continue
            atomcnt += 1
            if idx in sigIdx:
                sigIdx_general.append(idx)
                sigIdx_atom.append(atomcnt)
    return sigIdx_general, sigIdx_atom

def get_substruct(mol, atom_idx_list, radius = 3):
    # this function creates submolecules
    smiDic = {}#key, value: <substr in str, list of amap> / each of the amap elements are the indices of the significant atoms
    for r in range(2, radius)[::-1]:
        #can extract the submolecule consisting of all atoms within a radius of r of atom_idx
        for atom_idx in atom_idx_list:
            #print("atom_idx", atom_idx, Chem.MolToSmiles(mol))
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx)
            amap = {}#key, val = <significant atom index(different from whole index, order)>
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            subsmi = Chem.MolToSmiles(submol)

            if subsmi != "":
                #found the submolecule
                #break
                smiDic[subsmi] = amap.keys()
    return smiDic

#I found that the atommap attaches special bond characters (not all cases...) if they are inside the consecutive atoms
#this func insert the sign "-" between the indicies of the succesive atoms
def get_span_indicies(dicSubstrAmap):
    dicSpan = {}
    for substr, atomlist in dicSubstrAmap.items():
        spanlist = []
        prevIndex = -999
        for i, atomIndex in enumerate(atomlist):
            if atomIndex - prevIndex == 1:
                spanlist.append("-")

            spanlist.append(atomIndex)
            prevIndex = atomIndex
        dicSpan[substr] = spanlist
    return dicSpan

#the atommap of the rdkit calculates atom indicies only. this func changes it into a true SMILES indicies
def get_real_indicies(dicSpan, currSmiList, bond_score_include=True):
    twoChars = {"Al": 1, "Au": 1, "Ag": 1, "As": 1, "Ba": 1, "Be": 1, "Bi": 1, "Br": 1, "Ca": 1, "Cd": 1, "Cl": 1,
                "Co": 1, "Cr": 1, "Cu": 1, "Dy": 1, "Fe": 1, "Gd": 1, "Ge": 1, "In": 1, "Li": 1, "Mg": 1, "Mn": 1,
                "Mo": 1, "Na": 1, "Ni": 1, "Nd": 1, "Pb": 1, "Pt": 1, "Pd": 1, "Ru": 1, "Sb": 1, "Se": 1, "se": 1,
                "Si": 1, "Sn": 1, "Sr": 1, "Ti": 1, "Tl": 1, "Yb": 1, "Zn": 1, "Zr": 1}
    dicReal = {}
    for substr, spanlist in dicSpan.items():
        listtmp = []
        atomcnt = -1  # first atom is 0
        idx_span = 0
        includeFlag = False
        for idx, ele in enumerate(currSmiList):
            if ele in twoChars or ele.isalpha():  # or ele.isdigit():
                if ele == "@" or ele == "H":  # these are the same as the bond signals
                    continue
                atomcnt += 1
                if atomcnt == spanlist[idx_span]:
                    includeFlag = False
                    listtmp.append(idx)
                    idx_span += 1
                    if len(spanlist) == idx_span:
                        break
                    if spanlist[idx_span] == "-":
                        idx_span += 1
                        includeFlag = True
            if includeFlag and bond_score_include:
                listtmp.append(idx)
        dicReal[substr] = listtmp
    return dicReal

# this function updates dicSMA
#key, value: <substr in str, list of scores> / each of the scores are the sum of the atoms in the smart
def get_scores(dicReal, layer_weights, dicSMA):
    for substr, realIndicies in dicReal.items():
        score = 0.0
        for i in realIndicies:
            score += layer_weights[i]
        if substr in dicSMA:
            dicSMA[substr].append(score)
        else:
            dicSMA[substr] = [score]

#extract importnat structures using z-score over 2.58
def setThreshold(gmSMA):
    cnt = 0
    print(len(gmSMA))
    #get the key and value arrays from dict
    keylist, vallist = [], []
    for key, val in gmSMA.items():
        keylist.append(key)
        vallist.append(val)

    nparray = np.array(vallist)
    zscorearray = stats.zscore(nparray)

    return keylist, zscorearray
    
#extract importnat structures
def extSigSA(keylist, zscorearray):
    retSAs = []
    for i, zval in enumerate(zscorearray):
        if zval >= 2.58:
            retSAs.append(keylist[i])
    return retSAs
    
   
#############################
#main code starts
if not USE_DUMPED_DATA:
    IGweights1, teststr1 = pickle.load(open( "save_igtrain2", "rb" ))
    IGweights2, teststr2 = pickle.load(open( "save_igtest2", "rb" ))
    IGweights3, teststr3 = pickle.load(open( "save_igvalid2", "rb" ))

    #make dicSMA
    #key, value: <substr in str, list of scores> / each of the scores are the sum of the atoms in the smart
    dicSMA = {}
    layer1_sum = tf.math.reduce_sum(IGweights1, axis=2)#layer1_max shape = (whole_size, seq_len)
    layer2_sum = tf.math.reduce_sum(IGweights2, axis=2)
    layer3_sum = tf.math.reduce_sum(IGweights3, axis=2)
    layer_sum = tf.keras.layers.concatenate([layer1_sum, layer2_sum, layer3_sum], axis=0)

    teststrs = teststr1 + teststr2 + teststr3
    assert len(layer_sum) == len(teststrs)

    shared, notShared = 0, 0
    for i, currSmiList in enumerate(teststrs):
        currsmi = "".join(currSmiList)
        currmol = Chem.MolFromSmiles(currsmi)

        th_Until7thW = sorted(layer_sum[i])[-7]
        sigIdxList = idxForSignificants(th_Until7thW, layer_sum[i])
        sigIdx_general, sigIdx_atom = sigIdxAtom(sigIdxList, currSmiList)
        dicSubstrAmap = get_substruct(currmol, sigIdx_atom)#substrs of the current smi
        dicSpan = get_span_indicies(dicSubstrAmap)
        dicReal = get_real_indicies(dicSpan, currSmiList, bond_score_include=False)
        get_scores(dicReal, layer_sum[i], dicSMA)

    #calculate global mean score of every substrs
    gmSMA = {}
    for key, val in dicSMA.items():
        gmSMA[key] = np.mean(val)

    keylist, zscorearray = setThreshold(gmSMA)
    extractedSigSA = extSigSA(keylist, zscorearray)
    pickle.dump(extractedSigSA, open( "extractedSA2.pickle", "wb" ) )
else:
    extractedSigSA = pickle.load(open( "extractedSA2.pickle", "rb" ))
print(len(extractedSigSA))
    
#To compare extracted Significant SA with other SAs, load SAs from bioalerts
bioalert_subst = pickle.load(open( "nr-ar_bioalerts", "rb" ))

###########################
#this part is for checking only
dic0 = {}
act1, act0 = 0, 0
for idx, key in enumerate(bioalert_subst['Substructure in Molecule']):#molecule duplicated
    actlabel = bioalert_subst['Activity label'][idx]
    if actlabel == 1.0:
        act1 += 1
        dic0[Chem.MolToSmiles(key)] = 1
    else:
        act0 += 1
        if Chem.MolToSmiles(key) in dic0 and dic0[Chem.MolToSmiles(key)] == 1:
            dic0[Chem.MolToSmiles(key)] = 1
        else:
            dic0[Chem.MolToSmiles(key)] = 0
print("##############bioalerts#############")
print("positive:", act1, "negative:", act0, "unique smiles key:",len(dic0))
############################

#make substructure (mol class) list from bioalerts
subst_dics_bioalert = {}
subst_mols_bioalert = {}
for idx, key in enumerate(bioalert_subst["Substructure"]):
    subsmi = Chem.MolToSmiles(key)
    if subsmi not in subst_dics_bioalert:
        subst_dics_bioalert[subsmi] = 1
        subst_mols_bioalert[key] = subsmi

#substructures from bioalerts usually have similar structures => make uninque list
def rdkit_unique(subst_mols):
    subst_mols_unique = {}
    for key in subst_mols.keys():
        flag = False
        for key2 in subst_mols_unique.keys():
            if key == key2:
                contiune
            try:
                if key.HasSubstructMatch(key2):
                    flag = True
                    break
            except:
                flag = True
                break
        if not flag:
            subst_mols_unique[key] = subst_mols[key]
    print(len(subst_mols_unique))
    return subst_mols_unique

subst_mols_bioalertU = rdkit_unique(subst_mols_bioalert)


#make substructure (mol class) list from neural network
subst_dics_NN = {}
subst_mols_NN = {}
for subsmi in extractedSigSA:
    submol = Chem.MolFromSmarts(subsmi)
    if subsmi not in subst_dics_NN:
        subst_dics_NN[subsmi] = 1
        subst_mols_NN[submol] = subsmi

#subst_mols_NNU = rdkit_unique(subst_mols_NN) <= not required if the instances are too small

#compare substructures from NN with those from the bioalerts
cntShared = 0
for key, val in subst_mols_NN.items():
    flag = False
    for key2, val2 in subst_mols_bioalertU.items():
        if len(val) > len(val2):
            if key.HasSubstructMatch(key2):
                flag = True
                break
        else:
            if key2.HasSubstructMatch(key):
                flag = True
                break
    if flag:
        cntShared += 1
print(cntShared, len(subst_mols_NN), len(subst_mols_bioalertU))


#To compare extracted Significant SA with other SAs, load SAs from sarpy
subst_dics_sarpy = {}#this is required, since MOL class is hard to distinguish each other
subst_mols_sarpy = {}
f = open("nr_ar_sarpy.txt", "r")
lines = f.readlines()
for i, line in enumerate(lines):
    if i == 0:
        continue
    splitted = line.split("\t")
    if splitted[0] not in subst_dics_sarpy:
        subst_dics_sarpy[splitted[0]] = 1
        subst_mols_sarpy[Chem.MolFromSmarts(splitted[0])] = splitted[0]
f.close()

#=> make uninque list
subst_mols_sarpyU = rdkit_unique(subst_mols_sarpy)

#compare substructures from NN with those from the bioalerts
cntShared = 0
for key, val in subst_mols_NN.items():
    flag = False
    for key2, val2 in subst_mols_sarpyU.items():
        try:
            if key.HasSubstructMatch(key2):
                flag = True
                break
        except:
            continue
    if flag:
        cntShared += 1
print(cntShared, len(subst_mols_NN), len(subst_mols_sarpyU))
