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
            atomcnt += 1
            if idx in sigIdx:
                sigIdx_general.append(idx)
                sigIdx_atom.append(atomcnt)
    return sigIdx_general, sigIdx_atom

def get_substruct(mol, atom_idx_list, radius = 3):
    # this function creates submolecules
    smiList = {}
    for r in range(radius)[::-1]:
        #can extract the submolecule consisting of all atoms within a radius of r of atom_idx
        for atom_idx in atom_idx_list:
            print("atom_idx",atom_idx)
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx)
            amap = {}
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            smi = Chem.MolToSmiles(submol)

            if smi != "":
                #found the submolecule
                #break
                smiList[smi] = 1
    return list(smiList)

#due to the nature of our extracted substructure, we need to fill the hole
#if the atoms are 1 distance away (an atom is not included in the list) => fill the gap
#if the atoms are 2 distance away => discard the atom outlier
def idxMergeAtomLists(sigIdx_general, sigIdx_atom):
    if len(sigIdx_general) == 1:
        return sigIdx_general
    retlist = []
    twoChars = {"AL":1, "al":1, "Al":1, "AU":1, "au":1, "Au":1, "MG":1, "mg":1, "Mg":1, "ZN":1, "zn":1, "Zn":1, "CA":1, "ca":1, "Ca":1, "NA":1, "na":1, "Na":1, "CL":1, "cl":1, "Cl":1, "FE":1, "fe":1, "Fe":1, "BR":1, "br":1, "Br":1, "SI":1, "si":1, "Si":1}
    for idx, ele in enumerate(sigIdx_general):
        if idx == len(sigIdx_general) - 1:
            break
        if sigIdx_atom[idx + 1] - sigIdx_atom[idx] > 2:
            continue
        else:#if sigIdx_atom[idx + 1] - sigIdx_atom[idx] == 1 or sigIdx_atom[idx + 1] - sigIdx_atom[idx] == 2:
            if ele not in retlist:#becuase we add the next element ...
                retlist.append(ele)
            tmplist = []
            for i in range(sigIdx_general[idx] + 1, sigIdx_general[idx + 1]):
                tmplist.append(i)
            retlist.extend(tmplist)
            retlist.append(sigIdx_general[idx + 1])
    return retlist

#related to idxMergeAtomLists
#this func connects the original SMILES with the significant atoms
def atomListsIntoSmiFrags(mergedAtomList, currSmiList):
    retlist = []
    prevIndex = 0
    tmplist = []
    for i, ele in enumerate(mergedAtomList):
        if i != 0:
            if ele - prevIndex > 1:
                retlist.append("".join(tmplist))
                tmplist = []
        tmplist.append(currSmiList[ele])
        prevIndex = ele
        
    if tmplist != []:
        retlist.append("".join(tmplist))
    return retlist

attnweights, teststr = pickle.load(open( "save_igtest2", "rb" ))
bioalert_nr_ar_lbd = pickle.load(open( "nr-ar_bioalerts", "rb" ))

dic = {}
act1, act0 = 0, 0
for idx, key in enumerate(bioalert_nr_ar_lbd['Substructure in Molecule']):#molecule duplicated
    actlabel = bioalert_nr_ar_lbd['Activity label'][idx]
    if actlabel == 1.0:
        act1 += 1
        dic[Chem.MolToSmiles(key)] = 1
    else:
        act0 += 1
        if Chem.MolToSmiles(key) in dic and dic[Chem.MolToSmiles(key)] == 1:
            dic[Chem.MolToSmiles(key)] = 1
        else:
            dic[Chem.MolToSmiles(key)] = 0
print(act1, act0)
print(len(dic))

cnt, cntPos = 0, 0
for currSmiList in teststr:
    currsmi = "".join(currSmiList)
    if currsmi in dic:
        cnt += 1
        if dic[currsmi] == 1:
            cntPos += 1
print(cnt, cntPos)

dic = {}
for idx, key in enumerate(bioalert_nr_ar_lbd['Substructure in Molecule']):#molecule duplicated
    smikey = Chem.MolToSmiles(key)
    if smikey in dic:
        dic[smikey].append(idx)
    else:
        dic[smikey] = [idx]
print(len(dic))

def howManyAshareB(A, B):
    share, notshare = 0, 0
    for ele in A:
        if ele in B:
            share += 1
        else:
            notshare += 1
    return share, notshare, share + notshare


layer1_max = tf.math.reduce_max(attnweights, axis=2)#layer1_max shape = (whole_size, seq_len)
shared, notShared = 0, 0
for i, currSmiList in enumerate(teststr):
    currsmi = "".join(currSmiList)
    print("currsmi:",currsmi)
    currmol = Chem.MolFromSmiles(currsmi)
    if currsmi in dic:
        th_Until7thW = sorted(layer1_max[i])[-7]
        sigIdxList = idxForSignificants(th_Until7thW, layer1_max[i])
        sigIdx_general, sigIdx_atom = sigIdxAtom(sigIdxList, teststr[i])
        smarts_list = get_substruct(currmol, sigIdx_atom)        
        #sigIdx_atom_merged = idxMergeAtomLists(sigIdx_general, sigIdx_atom)
        #smarts_list = atomListsIntoSmiFrags(sigIdx_atom_merged, teststr[i])
        isSmartsIn = False
        for idx in dic[currsmi]:
            sigAtomsInMol = list(currmol.GetSubstructMatch(bioalert_nr_ar_lbd['Substructure'][idx]))
            substructure_smi = Chem.MolToSmiles(bioalert_nr_ar_lbd['Substructure'][idx])
            #print(sigAtomsInMol, Chem.MolToSmiles(bioalert_nr_ar_lbd['Substructure'][idx]))
            for smart in smarts_list:
                if smart.lower() in substructure_smi.lower():
                    isSmartsIn = True
                    print(smart, substructure_smi)
        if isSmartsIn:
            shared += 1
        else:
            notShared += 1
print(shared, notShared)
