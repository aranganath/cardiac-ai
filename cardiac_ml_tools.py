import glob, re, os
from typing import List
import numpy as np

def read_data_dirs(
        dirs_names : List[str] = ['/usr/workspace/hossain4/cardiac_ai/intracardiac_dataset/data_hearts_dd_0p2/'],
        verbose : int = 0) -> List[List[str]]:
    file_pairs = []

    for dir in dirs_names:
        all_files = sorted(glob.glob(dir + '/*.npy'))
        files_Vm = []
        files_pECG = []

        if verbose>0:
            print('Reading files...', end='')
        for file in all_files:
            if 'VmData' in file:
                files_Vm.append(file)
            if 'pECGData' in file:
                files_pECG.append(file)
        if verbose>0:
            print(' done.')

        if verbose > 0:
            print('len(files_pECG) : {}'.format(len(files_pECG)))
            print('len(files_Vm) : {}'.format(len(files_Vm)))

        for i in range(len(files_pECG)):
            VmName = files_Vm[i]
            VmName = VmName.replace('VmData', '')
            pECGName = files_pECG[i]
            pECGName = pECGName.replace('pECGData','')
            if pECGName == VmName:
                file_pairs.append([files_pECG[i], files_Vm[i]])
            else:
                print('Automatic sorted not matching, looking for pairs ...', end='')
                for j in range(len(files_Vm)):
                    VmName = files_Vm[j]
                    VmName = VmName.replace('VmData','')
                    if pECGName == VmName:
                        file_pairs.append([files_pECG[i], files_Vm[j]])
                print('done.')

    return file_pairs


def get_standard_leads(
        pECGnumpy : np.ndarray
        ) -> np.ndarray:
    

    ecg12aux = np.zeros((pECGnumpy.shape[0],12))
    WilsonLead = 0.33333333 * (pECGnumpy[:,0] + pECGnumpy[:,1] + pECGnumpy[:,2])
    ecg12aux[:,0] = pECGnumpy[:,1] - pECGnumpy[:,0]
    ecg12aux[:,1] = pECGnumpy[:,2] - pECGnumpy[:,0]
    ecg12aux[:,2] = pECGnumpy[:,2] - pECGnumpy[:,1]
    ecg12aux[:,3] = 1.5*(pECGnumpy[:,0] - WilsonLead)
    ecg12aux[:,4] = 1.5*(pECGnumpy[:,1] - WilsonLead)
    ecg12aux[:,5] = 1.5*(pECGnumpy[:,2] - WilsonLead)
    ecg12aux[:,6] = pECGnumpy[:,4] - WilsonLead
    ecg12aux[:,7] = pECGnumpy[:,5] - WilsonLead
    ecg12aux[:,8] = pECGnumpy[:,6] - WilsonLead
    ecg12aux[:,9] = pECGnumpy[:,7] - WilsonLead
    ecg12aux[:,10] = pECGnumpy[:,8] - WilsonLead
    ecg12aux[:,11] = pECGnumpy[:,9] - WilsonLead

    return ecg12aux

def get_activation_time(
        Vm : np.ndarray
        ) -> np.ndarray :
    
    actTime = []
    if Vm.shape[1] != 75:
        print('Error: Vm doesnot have 75 columns')
        return actTime
    
    for col in range(0,75,1):
        actTime.append(np.argmax(Vm[:,col]>0))
    actTime = np.asarray(actTime)
    actTime = np.reshape(actTime,(75,1))
    return actTime