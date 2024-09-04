import os
import torch
from torch.utils.data import Dataset
from typing import Tuple
from utils.datautil import fileReader, get_indices
from pdb import set_trace
import numpy as np

class TransformerDataset(Dataset):
    """
    Dataset class used for transformer models.
    
    """
    feat_dims = {
        "in_channels" : 12,
        "out_channels": 75,
    }
    def __init__(self,
        mode: str,
        subset:str,
        configs: dict,
        # VmData: torch.tensor,
        # ECGData: torch.tensor,
        # actTimeData: torch.tensor,
        # datInd: list,
        # enc_seq_len: int, 
        # dec_seq_len: int, 
        # target_seq_len: int
        ) -> None:

        """
        Args:

            data: tensor, the entire train, validation or test data sequence 
                        before any slicing. If univariate, data.size() will be 
                        [number of samples, number of variables]
                        where the number of variables will be equal to 1 + the number of
                        exogenous variables. Number of exogenous variables would be 0
                        if univariate.

            indices: a list of tuples. Each tuple has two elements:
                     1) the start index of a sub-sequence
                     2) the end index of a sub-sequence. 
                     The sub-sequence is split into src, trg and trg_y later.  

            enc_seq_len: int, the desired length of the input sequence given to the
                     the first layer of the transformer model.

            target_seq_len: int, the desired length of the target sequence (the output of the model)

            target_idx: The index position of the target variable in data. Data
                        is a 2D tensor
        """
        
        super().__init__()
        

        EncoderData, DecoderData, self.IntData = fileReader(configs['datapath'], configs["file"], configs["ratio"], mode)
        
        self.DecoderData = (DecoderData - torch.min(DecoderData))/(torch.max(DecoderData)-torch.min(DecoderData))
        self.EncoderData = (EncoderData - torch.min(EncoderData))/(torch.max(EncoderData) - torch.min(EncoderData))

        # self.datInd = datInd
        self.indices = get_indices(
            stop_position=self.EncoderData.shape[1],
            window_size=configs["start"],
            step_size=configs["step_size"]
        )
        self.configs = configs



    def __len__(self):
        return self.EncoderData.shape[0]

    def __getitem__(self, idx):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """
        # Get the first and last element of the i'th tuple in the list self.VmInds

        inp_seq = self.EncoderData[idx,:,:]
        tar_seq = self.DecoderData[idx,:,:]
        act_time = self.IntData[idx, :]
        src, trg = self.get_src_trg(
            inp_sequence= inp_seq,
            target_sequence = tar_seq,
            enc_seq_len=self.configs["start"],
        )
        return {
            "x": src.permute(1,0), 
            "y": trg,
            "act_time": act_time
        }
        
    def get_src_trg(
        self,
        inp_sequence: torch.Tensor,
        target_sequence: torch.Tensor,
        enc_seq_len: int,
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

        # encoder input
        src = inp_sequence[:enc_seq_len,:]
        trg= []
        for i in range(len(self.indices)):
            start_trg_idx, end_trg_idx = self.indices[i][0], self.indices[i][1]
            trg.append(target_sequence[start_trg_idx:end_trg_idx, :])
        trg = torch.stack(trg, axis = 0)
        return src, trg
    




    '''
        3 subsets
        1 -> pretrainer (5 datapoints) -> evaluate (5 points back to the network) (Unless this is done)
        2 -> Training (< 15995 datapoints)  -> evaluate (subset of 15995 -> 500) (Use the trained from step 1)
        3 -> Evaluate on 16000 - 16117

    
    '''