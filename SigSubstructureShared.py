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
    smiDic = {}#key, value: <substr in str, list of amap> / each of the amap elements are the indices of the significant atoms
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
def get_real_indicies(dicSpan, currSmiList):
    twoChars = {"AL": 1, "al": 1, "Al": 1, "AU": 1, "au": 1, "Au": 1, "MG": 1, "mg": 1, "Mg": 1, "ZN": 1, "zn": 1,
                "Zn": 1, "CA": 1, "ca": 1, "Ca": 1, "NA": 1, "na": 1, "Na": 1, "CL": 1, "cl": 1, "Cl": 1, "FE": 1,
                "fe": 1, "Fe": 1, "BR": 1, "br": 1, "Br": 1, "SI": 1, "si": 1, "Si": 1}
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
            if includeFlag:
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
    sigIdx_general, sigIdx_atom = sigIdxAtom(sigIdxList, currSmiList)
    dicSubstrAmap = get_substruct(currmol, sigIdx_atom)#substrs of the current smi
    dicSpan = get_span_indicies(dicSubstrAmap)
    dicReal = get_real_indicies(dicSpan, currSmiList)
    get_scores(dicReal, layer1_sum[i], dicSMA)

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

