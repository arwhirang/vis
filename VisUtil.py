from rdkit import Chem
from scipy import stats

#######################################
########Extract SA code from NN########
#######################################
#pick the significant indicies that have high enough weights above certain threshold (7th elements from the biggest)
def idxForSignificants(th_, aWeight):
    retlist = []
    for idx, ele in enumerate(aWeight):
        if ele >= th_:
            retlist.append(idx)
    return retlist
    
    
#Since atoms are our main concern, separate the indicies of atoms from the significant indicies
def sigIdxAtom(sigIdx, currSmiList):
    sigIdx_general, sigIdx_atom = [], []
    twoChars = {"Al": 1, "Au": 1, "Ag": 1, "As": 1, "Ba": 1, "Be": 1, "Bi": 1, "Br": 1, "Ca": 1, "Cd": 1, "Cl": 1,
                "Co": 1, "Cr": 1, "Cu": 1, "Dy": 1, "Fe": 1, "Gd": 1, "Ge": 1, "In": 1, "Li": 1, "Mg": 1, "Mn": 1,
                "Mo": 1, "Na": 1, "Ni": 1, "Nd": 1, "Pb": 1, "Pt": 1, "Pd": 1, "Ru": 1, "Sb": 1, "Se": 1, "se": 1,
                "Si": 1, "Sn": 1, "Sr": 1, "Ti": 1, "Tl": 1, "Yb": 1, "Zn": 1, "Zr": 1}
    atomcnt = -1  # first atom is 0
    for idx, ele in enumerate(currSmiList):
        if ele in twoChars or ele.isalpha():
            if ele == "@" or ele == "H":  # these are treated as the same as the bond signals
                continue
            atomcnt += 1
            if idx in sigIdx:
                sigIdx_general.append(idx)
                sigIdx_atom.append(atomcnt)
    return sigIdx_general, sigIdx_atom


#creates submolecules from the significant atoms using rdkit
#also, create orimol_atomI tuple. orimol_atomI is for the comparison part
def get_substruct(mol, atom_idx_list, radius=3):
    subsmiDic = {}     # key, value: <substr in str, list of amap> / each of the amap elements are the indices of the significant atoms
    orimol_atomI = ()  # (orimol, list of amap) / each of the amap elements are the indices of the significant atoms
    for r in range(1, radius)[::-1]:
        # can extract the submolecule consisting of all atoms within a radius of r of atom_idx
        for atom_idx in atom_idx_list:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, r, atom_idx)
            amap = {}  # key, val = <atom index prime(different from whole index), order>
            submol = Chem.PathToSubmol(mol, env, atomMap=amap)
            subsmi = Chem.MolToSmiles(submol)

            if subsmi != "":# found the submolecule
                tmpAmapList = list(amap.keys())

                subsmiDic[subsmi] = tmpAmapList
                orimol_atomI = (mol, tmpAmapList)
    return subsmiDic, orimol_atomI
    
    
# the atommap of the rdkit calculates atom indicies while integrated gradients are based on SMILES.
# this func changes the atommap indicies into a SMILES indicies 
def get_real_indicies(dicSubstrAmap, currSmiList):
    #certain atoms consists of two characters
    twoChars = {"Al": 1, "Au": 1, "Ag": 1, "As": 1, "Ba": 1, "Be": 1, "Bi": 1, "Br": 1, "Ca": 1, "Cd": 1, "Cl": 1,
                "Co": 1, "Cr": 1, "Cu": 1, "Dy": 1, "Fe": 1, "Gd": 1, "Ge": 1, "In": 1, "Li": 1, "Mg": 1, "Mn": 1,
                "Mo": 1, "Na": 1, "Ni": 1, "Nd": 1, "Pb": 1, "Pt": 1, "Pd": 1, "Ru": 1, "Sb": 1, "Se": 1, "se": 1,
                "Si": 1, "Sn": 1, "Sr": 1, "Ti": 1, "Tl": 1, "Yb": 1, "Zn": 1, "Zr": 1}
    dicReal = {}
    for substr, atomlist in dicSubstrAmap.items():
        listtmp = []
        atomcnt = -1  # first atom is 0
        idx_span = 0
        for idx, ele in enumerate(currSmiList):
            if ele in twoChars or ele.isalpha():
                if ele == "@" or ele == "H":  # these are treated as the same as the bond signals
                    continue
                atomcnt += 1
                if atomcnt == atomlist[idx_span]:
                    listtmp.append(idx)
                    idx_span += 1
                    # error.. maybe
                    if len(atomlist) == idx_span:
                        print("fix the error plz")
                        break

        dicReal[substr] = listtmp
    return dicReal


# get_scores function updates dicSAs and dicOriMap
# dicSAs = key, value: <substr in str, list of scores> / each of the scores are the sum of the atoms in the smart
# dicOriMap = key, value: <substr in str, [orimol, list of amap]>
def get_scores(dicReal, layer_weights, orimol_atomI, dicSAs, dicOriAmap):
    for substr, realIndicies in dicReal.items():
        score = 0.0
        for i in realIndicies:
            score += layer_weights[i]
        if substr in dicSAs:
            dicSAs[substr].append(score)
            dicOriAmap[substr].append(orimol_atomI)
        else:
            dicSAs[substr] = [score]
            dicOriAmap[substr] = [orimol_atomI]


# make the score list into z-score arrays
def getZscores(gmSMA):
    cnt = 0
    # get the key and value arrays from dict
    keylist, vallist = [], []
    for key, val in gmSMA.items():
        keylist.append(key)
        vallist.append(val)

    nparray = np.array(vallist)
    zscorearray = stats.zscore(nparray)
    return keylist, zscorearray


# extract importnat structures that contain high z-scores 
def extSigSA(keylist, zscorearray, dicOriAmap, ZSCORE):
    retSADict = {}  # key, val = <substr in str, list of (orimol, list of amap)>
    for i, zval in enumerate(zscorearray):
        if zval >= ZSCORE:
            retSADict[keylist[i]] = dicOriAmap[keylist[i]]
    return retSADict
    
    
#######################################
####Extract SA code from Bioalerts#####
#######################################
def load_bioalert(bioalert_subst):
    # make substructure (mol class) list from bioalerts
    subst_dics_bioalert = {}  # this is for making unique subst dict
    subst_mols_bioalert = {}  # key, val = <substructure mol, original mol>
    for key in bioalert_subst.T.items():
        submol = key[1]["Substructure"]
        orimol = key[1]["Substructure in Molecule"]#actually, this will not be used. for later use
        subsmi = Chem.MolToSmiles(submol)
        if subsmi not in subst_dics_bioalert:
            subst_dics_bioalert[subsmi] = 1
            subst_mols_bioalert[submol] = orimol
    return subst_mols_bioalert


# substructures from bioalerts usually have similar structures => make uninque list
def rdkit_unique(subst_mols):
    subst_mols_unique = {}
    for key, val in subst_mols.items():               # key, val = <substructure mol, original mol>
        flag = False
        for key2, val2 in subst_mols_unique.items():  # key, val = <substructure mol, original mol>
            if key == key2:
                contiune
            if val2.HasSubstructMatch(key):
                flag = True
                break
        if not flag:
            subst_mols_unique[key] = subst_mols[key]
    print("uniq nums bioalerts:", len(subst_mols_unique))
    return subst_mols_unique

    
#######################################
######Extract SA code from Sarpy#######
#######################################
def load_sarpy(fname):
    # To compare extracted Significant SA with other SAs, load SAs from sarpy
    subst_dics_sarpy = {}  # this is required, since MOL class is hard to distinguish each other
    subst_mols_sarpy = {}  # key, val = <substructure mol, substructure smiles>
    f = open("refData/" + fname + ".txt", "r")
    lines = f.readlines()
    for i, line in enumerate(lines):
        if i == 0:
            continue
        splitted = line.split("\t")
        if splitted[0] not in subst_dics_sarpy:
            subst_dics_sarpy[splitted[0]] = 1
            subst_mols_sarpy[Chem.MolFromSmarts(splitted[0])] = splitted[0]
    f.close()
    return subst_mols_sarpy


#######################################
###########Comparison codes############
#######################################
def _lstOrimolHasSubst(listOfOri, subMol):
    for mol, lstAtom in listOfOri:
        if mol.HasSubstructMatch(subMol):
            return (mol, lstAtom)
    return None


#Since the NN model is based on a significant atoms, check if the sig atoms are inside the substructure from other methods
def _isHitTheSame(orimolTuple, subMol):
    hit_ats = list(orimolTuple[0].GetSubstructMatch(subMol))
    for hitatIndex in hit_ats:
        if hitatIndex in orimolTuple[1]:
            return True
    return False


#comparison code!
def compareWithAtomUnits(subst_mols_NN, subst_mols_statistics):
    # compare substructures from NN with those from the bioalerts
    cntShared = 0
    for key, val_lst in subst_mols_NN.items():  # key, val = <substructure mol, list of (original mol, list of amap)>
        orimolTuple = None
        for key2, val2 in subst_mols_statistics.items():  # key, val = <substructure mol, substructure smiles>
            orimolTuple = _lstOrimolHasSubst(val_lst, key2)
            if orimolTuple is not None:
                break
        if orimolTuple:
            if _isHitTheSame(orimolTuple, key2):
                cntShared += 1
    print("shared with: ", cntShared, len(subst_mols_NN), len(subst_mols_statistics))

