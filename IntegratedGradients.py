################################################################
# Implemented by Naozumi Hiranuma (hiranumn@uw.edu)            #
#                                                              #
# Keras-compatible implmentation of Integrated Gradients       # 
# proposed in "Axiomatic attribution for deep neuron networks" #
# (https://arxiv.org/abs/1703.01365).                          #
#                                                              #
# Keywords: Shapley values, interpretable machine learning     #
################################################################

#from __future__ import division, print_function
import numpy as np
import sys
import tensorflow as tf
#from keras.models import Model, Sequential

'''
Integrated gradients approximates Shapley values by integrating partial
gradients with respect to input features from reference input to the
actual input. The following class implements the paper "Axiomatic attribution
for deep neuron networks".
'''
class integrated_gradients:
    # model: Keras model that you wish to explain.
    # outchannels: In case the model are multi tasking, you can specify which output you want explain .
    def __init__(self, model, outchannels=[], verbose=1):
        self.model = model
        #load input tensors
        self.input_tensors = []
        for i in self.model.inputs:
            self.input_tensors.append(i)
        # The learning phase flag is a bool tensor (0 = test, 1 = train)
        # to be passed as input to any Keras function that uses 
        # a different behavior at train time and test time.
        #self.input_tensors.append(K.learning_phase())
        
        #If outputchanels are specified, use it.
        #Otherwise evalueate all outputs.
        self.outchannels = outchannels
        if len(self.outchannels) == 0: 
            if verbose:
                print("Evaluated output channel (0-based index): All")
            print(self.model.output.shape)
            self.outchannels = range(self.model.output.shape[1]._value)
        else:
            if verbose: 
                print("Evaluated output channels (0-based index):")
                print(','.join([str(i) for i in self.outchannels]))
                
        #Build gradient functions for desired output channels.
        self.get_gradients = {}
        if verbose:
            print("Building gradient functions")
        
        # Evaluate over all requested channels.
        for c in self.outchannels:
            # Get tensor that calculates gradient    
            gradients = self.model.optimizer.get_gradients(self.model.output[:, c], self.model.input)
            # Build computational graph that computes the tensors given inputs
            self.get_gradients[c] = tf.keras.backend.function(inputs=self.input_tensors, outputs=gradients)
            
            # This takes a lot of time for a big model with many tasks.
            # So lets print the progress.
            if verbose:
                sys.stdout.write('\r')
                sys.stdout.write("Progress: "+str(int((c+1)*1.0/len(self.outchannels)*1000)*1.0/10)+"%")
                sys.stdout.flush()
        # Done
        if verbose:
            print("\nDone.")
            
                
    '''
    Input: sample to explain, channel to explain
    Optional inputs:
        - reference: reference values (defaulted to 0s).
        - steps: # steps from reference values to the actual sample (defualted to 50).
    Output: list of numpy arrays to integrated over.
    '''
    def explain(self, sample, outc=0, reference=False, num_steps=50, verbose=0):
        # Each element for each input stream.
        samples = []
        numsteps = []
        step_sizes = []
        
        print(outc, self.outchannels)
        # If multiple inputs are present, feed them as list of np arrays. 
        if isinstance(sample, list):
            #If reference is present, reference and sample size need to be equal.
            if reference != False: 
                assert len(sample) == len(reference)
            for i in range(len(sample)):
                if reference == False:
                    _output = integrated_gradients.linearly_interpolate(sample[i], False, num_steps)
                else:
                    _output = integrated_gradients.linearly_interpolate(sample[i], reference[i], num_steps)
                samples.append(_output[0])
                numsteps.append(_output[1])
                step_sizes.append(_output[2])
        
        # Or you can feed just a single numpy arrray. 
        elif isinstance(sample, np.ndarray):
            _output = integrated_gradients.linearly_interpolate(sample, reference, num_steps)
            samples.append(_output[0])
            numsteps.append(_output[1])
            step_sizes.append(_output[2])
            
        # Desired channel must be in the list of outputchannels
        assert outc in self.outchannels
        if verbose:
            print("Explaning the "+str(self.outchannels[outc])+"th output.")
            
        # For tensorflow backend
        _input = []
        for s in samples:
            _input.append(s)
        _input.append(0)
        gradients = self.get_gradients[outc](_input)
        
        explanation = []
        for i in range(len(gradients)):
            _temp = np.sum(gradients[i], axis=0)
            explanation.append(np.multiply(_temp, step_sizes[i]))
           
        # Format the return values according to the input sample.
        if isinstance(sample, list):
            return explanation
        elif isinstance(sample, np.ndarray):
            return explanation[0]
        return -1

    
    '''
    Input: numpy array of a sample
    Optional inputs:
        - reference: reference values (defaulted to 0s).
        - steps: # steps from reference values to the actual sample.
    Output: list of numpy arrays to integrate over.
    '''
    @staticmethod
    def linearly_interpolate(sample, reference=False, num_steps=50):
        # Use default reference values if reference is not specified
        if reference is False:
            reference = np.zeros(sample.shape);

        # Reference and sample shape needs to match exactly
        assert sample.shape == reference.shape

        # Calcuated stepwise difference from reference to the actual sample.
        ret = np.zeros(tuple([num_steps] + [i for i in sample.shape]))
        for s in range(num_steps):
            ret[s] = reference + (sample - reference)*(s*1.0/num_steps)
        return ret, num_steps, (sample - reference)*(1.0/num_steps)
