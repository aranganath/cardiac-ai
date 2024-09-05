import torch
from typing import Optional, Any, Union, Callable, Tuple, List
import os
import glob
import numpy as np
import re
from tqdm import tqdm
from pdb import set_trace


def generate_square_subsequent_mask(dim1: int, dim2: int) -> torch.Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.
    Modified from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html

    Args:

        dim1: int, for both src and tgt masking, this must be target sequence
              length

        dim2: int, for src masking this must be encoder sequence length (i.e. 
              the length of the input sequence to the model), 
              and for tgt masking, this must be target sequence length 


    Return:

        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float('-inf'), diagonal=1)


def get_src_trg(
    self,
    sequence: torch.Tensor, 
    enc_seq_len: int, 
    target_seq_len: int
    ) -> Tuple[torch.tensor, torch.tensor, torch.tensor]:

    """
    Generate the src (encoder input), trg (decoder input) and trg_y (the target)
    sequences from a sequence. 

    Args:

        sequence: tensor, a 1D tensor of length n where 
                n = encoder input length + target sequence length  

        enc_seq_len: int, the desired length of the input to the transformer encoder

        target_seq_len: int, the desired length of the target sequence (the 
                        one against which the model output is compared)

    Return: 

        src: tensor, 1D, used as input to the transformer model

        trg: tensor, 1D, used as input to the transformer model

        trg_y: tensor, 1D, the target sequence against which the model output
            is compared when computing loss. 
    
    """
    #print("Called dataset.TransformerDataset.get_src_trg")
    assert len(sequence) == enc_seq_len + target_seq_len, "Sequence length does not equal (input length + target length)"
    
    #print("From data.TransformerDataset.get_src_trg: sequence shape: {}".format(sequence.shape))

    # encoder input
    src = sequence[:enc_seq_len] 
    
    # decoder input. As per the paper, it must have the same dimension as the 
    # target sequence, and it must contain the last value of src, and all
    # values of trg_y except the last (i.e. it must be shifted right by 1)
    trg = sequence[enc_seq_len-1:len(sequence)-1]

    #print("From data.TransformerDataset.get_src_trg: trg shape before slice: {}".format(trg.shape))

    trg = trg[:, 0]

    #print("From data.TransformerDataset.get_src_trg: trg shape after slice: {}".format(trg.shape))

    if len(trg.shape) == 1:

        trg = trg.unsqueeze(-1)

        #print("From data.TransformerDataset.get_src_trg: trg shape after unsqueeze: {}".format(trg.shape))

    
    assert len(trg) == target_seq_len, "Length of trg does not match target sequence length"

    # The target sequence against which the model output will be compared to compute loss
    trg_y = sequence[-target_seq_len:]

    #print("From data.TransformerDataset.get_src_trg: trg_y shape before slice: {}".format(trg_y.shape))

    # We only want trg_y to consist of the target variable not any potential exogenous variables
    trg_y = trg_y[:, 0]

    #print("From data.TransformerDataset.get_src_trg: trg_y shape after slice: {}".format(trg_y.shape))

    assert len(trg_y) == target_seq_len, "Length of trg_y does not match target sequence length"

    return src, trg, trg_y.squeeze(-1) # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len] 



# All additional files we will be using

# funtion to read the data
def read_data_dirs(
        dirs_names : List[str] = ['../intracardiac_dataset/data_hearts_dd_0p2/'],
        verbose : int = 0) -> List[List[str]]:
    """
    Read the numpy files in the given directories.
    Returns a list of file pairs ECG/Vm.
    
    Parameters
    ----------
    dirs_names : List[str]
        List of directories containing the data.
    verbose : int
        Verbosity level.
    
    Returns
    -------
    file_pairs : List[List[str]]
        List of file pairs.
    """
    file_pairs = []
    
    for dir in dirs_names:    
        all_files = sorted(glob.glob(dir + '/*.npy'))
        files_Vm=[]
        files_pECG=[]
        
        if verbose > 0:
            print('Reading files...',end='')
        for file in all_files:
            if 'VmData' in file:
                files_Vm.append(file)
            if 'pECGData' in file:
                files_pECG.append(file)
        if verbose > 0:        
            print(' done.')
        
        if verbose > 0:
            print('len(files_pECG) : {}'.format(len(files_pECG)))
            print('len(files_Vm) : {}'.format(len(files_Vm)))
        
        for i in range(len(files_pECG)):  
            VmName =  files_Vm[i]
            VmName = VmName.replace('VmData', '')
            pECGName =  files_pECG[i]
            pECGName = pECGName.replace('pECGData', '')            
            if pECGName == VmName :
                file_pairs.append([files_pECG[i], files_Vm[i]])
            else:
                print('Automatic sorted not matching, looking for pairs ...',end='')
                for j in range(len(files_Vm)):
                    VmName =  files_Vm[j]
                    VmName = VmName.replace('VmData', '')
                    if pECGName == VmName :
                        file_pairs.append([files_pECG[i], files_Vm[j]])
                print('done.')       
    return file_pairs


# function to transform the data
def get_standard_leads(
        pECGnumpy : np.ndarray
    ) -> np.ndarray :
    """
    Get the standard 12-lead from the 10-lead ECG.
    
    Parameters
    ----------
    pECGnumpy : np.ndarray
        10-lead ECG.
        
    Returns
    -------
    ecg12aux : np.ndarray
        12-lead ECG.
    """
    # pECGnumpy  : RA LA LL RL V1 V2 V3 V4 V5 V6
    # ecg12aux : i, ii, iii, avr, avl, avf, v1, v2, v3, v4, v5, v6
    ecg12aux = np.zeros((pECGnumpy.shape[0],12))
    WilsonLead = 0.33333333 * (pECGnumpy[:,0] + pECGnumpy[:,1] + pECGnumpy[:,2])
    # Lead I: LA - RA
    ecg12aux[:,0] = pECGnumpy[:,1] - pECGnumpy[:,0]
    # Lead II: LL - RA
    ecg12aux[:,1] = pECGnumpy[:,2] - pECGnumpy[:,0]
    # Lead III: LL - LA
    ecg12aux[:,2] = pECGnumpy[:,2] - pECGnumpy[:,1]
    # Lead aVR: 3/2 (RA - Vw)
    ecg12aux[:,3] = 1.5*(pECGnumpy[:,0] - WilsonLead)
    # Lead aVL: 3/2 (LA - Vw)
    ecg12aux[:,4] = 1.5*(pECGnumpy[:,1] - WilsonLead)
    # Lead aVF: 3/2 (LL - Vw)
    ecg12aux[:,5] = 1.5*(pECGnumpy[:,2] - WilsonLead)
    # Lead V1: V1 - Vw
    ecg12aux[:,6] = pECGnumpy[:,4] - WilsonLead
    # Lead V2: V2 - Vw
    ecg12aux[:,7] = pECGnumpy[:,5] - WilsonLead
    # Lead V3: V3 - Vw
    ecg12aux[:,8] = pECGnumpy[:,6] - WilsonLead
    # Lead V4: V4 - Vw
    ecg12aux[:,9] = pECGnumpy[:,7] - WilsonLead
    # Lead V5: V5 - Vw
    ecg12aux[:,10] = pECGnumpy[:,8] - WilsonLead
    # Lead V6: V6 - Vw
    ecg12aux[:,11] = pECGnumpy[:,9] - WilsonLead

    return ecg12aux

# funtion to get the activation time
def get_activation_time(
        Vm : np.ndarray
    ) -> np.ndarray :
    """
    Get the activation time from the Vm.
    
    Parameters
    ----------
    Vm : np.ndarray
        Vm.
        
    Returns
    -------
    actTime : np.ndarray
        Activation time.
    """
    actTime = []
    # check that Vm has 75 columns
    if Vm.shape[1] != 75:
        print('Error: Vm does not have 75 columns')
        return actTime
    for col in range(0,75,1):
        actTime.append(np.argmax(Vm[:,col]>0))
    actTime = np.asarray(actTime)
    actTime = np.reshape(actTime,(75,1))
    return actTime

def fileReader(path: str, finalInd: int, train_test_ratio: float, mode: str):
    '''
    Args:
        path: Path where the data is residing at the moment
    '''

    # Now, let's load the data itself
    files = []
    regex = r'data_hearts_dd_0p2*'
    pECGTrainData, VmTrainData, pECGValData, VmValData, actTimeTrain, actTimeVal  = [], [], [], [], [], []

    for x in os.listdir(path):
        if re.match(regex, x):
            files.append(path + x)

    data_dirs = read_data_dirs(files)
    finalInd = min(finalInd,16000)
    if mode in ("train", "val"):
        indices = list(range(0,finalInd))
        data_dirs = [data_dirs[index] for index in indices]
        trainIndices = set(np.random.permutation(int(train_test_ratio*len(data_dirs))))
        for i, (pECGData_file, VmData_file) in enumerate(data_dirs):
            if i in trainIndices:
                with open(pECGData_file, 'rb') as f:
                    pECGTrainData.append(get_standard_leads(np.load(f)))
                with open(VmData_file, 'rb') as f:
                    VmTrainData.append(np.load(f))
                    actTimeTrain.append(get_activation_time(VmTrainData[-1]).squeeze(1))
            
            else:
                with open(pECGData_file, 'rb') as f:
                    pECGValData.append(get_standard_leads(np.load(f)))
                
                with open(VmData_file, 'rb') as f:
                    VmValData.append(np.load(f))
                    actTimeVal.append(get_activation_time(VmValData[-1]).squeeze(1))

        
        
        VmTrainData = np.stack(VmTrainData, axis = 0)
        pECGTrainData = np.stack(pECGTrainData, axis=0)
        VmValData = np.stack(VmValData, axis = 0)
        pECGValData = np.stack(pECGValData, axis = 0)
        actTimeTrain = np.stack(actTimeTrain, axis = 0)
        actTimeVal= np.stack(actTimeVal, axis = 0)
        return torch.from_numpy(pECGTrainData), torch.from_numpy(VmTrainData),  torch.from_numpy(actTimeTrain)
    
    elif mode == "test":
        pECGTestData, VmTestData, actTimeTest = [], [], []
        indices = set(list(range(16000, 16117)))
        data_dirs = [data_dirs[index] for index in indices]
        for i, (pECGData_file, VmData_file) in enumerate(tqdm(data_dirs, desc='Loading datafiles ')):
            
            with open(pECGData_file, 'rb') as f:
                pECGTestData.append(get_standard_leads(np.load(f)))
            
            with open(VmData_file, 'rb') as f:
                VmTestData.append(np.load(f))
                actTimeTest.append(get_activation_time(VmTestData[-1]).squeeze(1))

        
    
        VmTestData = np.stack(VmTestData, axis = 0)
        pECGTestData = np.stack(pECGTestData, axis = 0)
        actTimeTest = np.stack(actTimeTest, axis = 0)
        return torch.from_numpy(pECGTestData), torch.from_numpy(VmTestData), torch.from_numpy(actTimeTest)    


def get_indices(stop_position:int, window_size: int, step_size: int) -> list:
    """
    Produce all the start and end index positions that is needed to produce
    the sub-sequences. 

    Returns a list of tuples. Each tuple is (start_idx, end_idx) of a sub-
    sequence. These tuples should be used to slice the dataset into sub-
    sequences. These sub-sequences should then be passed into a function
    that slices them into input and target sequences. 
    
    Args:
        num_obs (int): Number of observations (time steps) in the entire 
                       dataset for which indices must be generated, e.g. 
                       len(data)

        window_size (int): The desired length of each sub-sequence. Should be
                           (input_sequence_length + target_sequence_length)
                           E.g. if you want the model to consider the past 100
                           time steps in order to predict the future 50 
                           time steps, window_size = 100+50 = 150

        step_size (int): Size of each step as the data sequence is traversed 
                         by the moving window.
                         If 1, the first sub-sequence will be [0:window_size], 
                         and the next will be [1:window_size+1].

    Return:
        indices: a list of tuples
    """
    
    subseq_first_idx, subseq_last_idx = 0, window_size
    
    datInd = []
    
    while subseq_last_idx <= stop_position:

        datInd.append((subseq_first_idx, subseq_last_idx))
    
        subseq_first_idx += step_size
        
        subseq_last_idx += step_size
    
    # if datInd[-1][1] != stop_position:
    #     datInd.append((subseq_first_idx, stop_position))

    return datInd

def fileReaderForActivation(path: str, dataInd: int):
    '''
    Load the data and get the activation for the output
    '''
    files = []

    regex = r'data_hearts_dd_0p2*'
    pECGTrainData, ActivationTrainData, pECGTestData, VmTestData  = [], [], [], []
    for x in os.listdir(path):
        if re.match(regex, x):
            files.append(path + x)
    
    data_dirs = read_data_dirs(files)[:dataInd]

    trainLength = int(0.8*len(data_dirs))

    for i, (pECGData_file, VmData_file) in enumerate(tqdm(data_dirs, desc='Loading datafiles ')):
        if i < trainLength:
            with open(pECGData_file, 'rb') as f:
                pECGTrainData.append(get_standard_leads(np.load(f)))
            with open(VmData_file, 'rb') as f:
                ActivationTrainData.append(np.argmax(np.load(f), axis = 0))
        
        else:
            with open(pECGData_file, 'rb') as f:
                pECGTestData.append(get_standard_leads(np.load(f)))
            
            with open(VmData_file, 'rb') as f:
                VmTestData.append(np.load(f))
        
    
    ActivationTrainData = np.stack(ActivationTrainData, axis = 0)
    pECGTrainData = np.stack(pECGTrainData, axis=0)
    VmTestData = np.stack(VmTestData, axis = 0)
    pECGTestData = np.stack(pECGTestData, axis = 0)
    return torch.from_numpy(ActivationTrainData), torch.from_numpy(pECGTrainData), torch.from_numpy(VmTestData), torch.from_numpy(pECGTestData)


# funtion to get the activation time
def get_activation_time(
        Vm : np.ndarray
    ) -> np.ndarray :
    """
    Get the activation time from the Vm.
    
    Parameters
    ----------
    Vm : np.ndarray
        Vm.
        
    Returns
    -------
    actTime : np.ndarray
        Activation time.
    """
    actTime = []
    # check that Vm has 75 columns
    if Vm.shape[1] != 75:
        print('Error: Vm does not have 75 columns')
        return actTime
    for col in range(0,75,1):
        actTime.append(np.argmax(Vm[:,col]>0))
    actTime = torch.from_numpy(np.asarray(actTime))
    actTime = actTime.reshape((75,1))
    return actTime