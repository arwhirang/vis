import pickle
import numpy as np
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem
from IPython.display import SVG
from rdkit.Chem.Draw import MolToImage
from rdkit.Chem import Draw
import tensorflow as tf

def idxForSignificants(th_, aWeight):
    retlist = []
    for idx, ele in enumerate(aWeight):
        if ele >= th_:
            retlist.append(idx)
    return retlist

def sigIdxAtom(sigIdx, currSmiList):
    sigIdx_general, sigIdx_atom = [], []
    twoChars = {"AL":1, "al":1, "Al":1, "AU":1, "au":1, "Au":1, "MG":1, "mg":1, "Mg":1, "ZN":1, "zn":1, "Zn":1, "CA":1, "ca":1, "Ca":1, "NA":1, "na":1, "Na":1, "CL":1, "cl":1, "Cl":1, "FE":1, "fe":1, "Fe":1, "BR":1, "br":1, "Br":1, "SI":1, "si":1, "Si":1}
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
    smiDic = {}
    for r in range(2, radius)[::-1]:
        #can extract the submolecule consisting of all atoms within a radius of r of atom_idx
        for atom_idx in atom_idx_list:
            #print("atom_idx",atom_idx)
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx)
            amap = {}#key, val = <significant atom index(different from whole index, order)>
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            subsmi = Chem.MolToSmiles(submol)

            if subsmi != "":
                #found the submolecule
                #break
                smiDic[subsmi] = amap.keys()
    return smiDic

# this function updates dicSMA
#key, value: <substr in str, list of scores> / each of the scores are the sum of the atoms in the smart
def get_scores(substr_dic, currsmi, layer_weights, dicSMA):
    for substr in substr_dic.keys():
        start = get_start(substr, currsmi)
        if start == -1:
            print("error!", substr, currsmi)
        score = 0.0
        for i in range(start, start + len(substr)):
            score += layer_weights[i]
        if substr in dicSMA:
            dicSMA[substr].append(score)
        else:
            dicSMA[substr] = [score]

IGweights, teststr = pickle.load(open( "save_igtest2", "rb" ))
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
print("", act1, act0)
print(len(dic0))
############################



#make dicSMA
#key, value: <substr in str, list of scores> / each of the scores are the sum of the atoms in the smart
dicSMA = {}
layer1_sum = tf.math.reduce_sum(IGweights, axis=2)#layer1_max shape = (whole_size, seq_len)
shared, notShared = 0, 0
for i, currSmiList in enumerate(teststr):
    currsmi = "".join(currSmiList)
    currmol = Chem.MolFromSmiles(currsmi)

    th_Until7thW = sorted(layer1_sum[i])[-7]
    sigIdxList = idxForSignificants(th_Until7thW, layer1_sum[i])
    sigIdx_general, sigIdx_atom = sigIdxAtom(sigIdxList, teststr[i])
    substrDic = get_substruct(currmol, sigIdx_atom)#substrs of the current smi
    get_scores(substr_list, currsmi, layer1_sum[i], dicSMA)

#calculate global mean score of every substrs
gmSMA = {}
for key, val in dicSMA.items():
    gmSMA[key] = mean(val)
    print(gmSMA[key])
#extract importnat structures

#compare

"""

dic_bioalert = {}
for idx, key in enumerate(bioalert_subst["Substructure"]):
    actlabel = bioalert_subst['Activity label'][idx]
    subsmi = Chem.MolToSmiles(key)
    if subsmi in dic_bioalert:
        dic_bioalert[subsmi].append(idx)
    else:
        dic_bioalert[subsmi] = [idx]
print(len(dic_bioalert))


    isSmartsIn = False
    for idx in dic[currsmi]:
        sigAtomsInMol = list(currmol.GetSubstructMatch(bioalert_subst['Substructure'][idx]))
        substructure_smi = Chem.MolToSmiles(bioalert_subst['Substructure'][idx])
        #print(sigAtomsInMol, Chem.MolToSmiles(bioalert_nr_ar_lbd['Substructure'][idx]))
        for smart in smarts_list:
            if smart.lower() in substructure_smi.lower():
                isSmartsIn = True
                print("currsmi:",currsmi)
                print(smart, substructure_smi)
    if isSmartsIn:
        shared += 1
    else:
        notShared += 1
print(shared, notShared)
"""

