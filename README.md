# vis
repository code for the ISMB poster "Quantitative evaluation of structural alerts extracted from deep learning QSAR models"  

TOX21 data can be downloaded from the following url: http://www.dna.bio.keio.ac.jp/smiles/  
We added the NR-AR domain data for demonstration.

refData folder contains sarpy and bioalerts results. We added the NR-AR domain results for demonstration. Since sarpy and bioalerts are independant libraries and they contain a lot of hyperparameters, we request users to make the results by themselves.  

saved folder contains the IG scores from the Neural Network results  

## step by step  
1. run the fp2vec code to train neural network and save the integrated gradients results  
2. run the SigSubstructureShared.py code to extract significant substructures and compare them with statistical results
