import os
import torch
from torch.utils.data import Dataset
from typing import Tuple
from pdb import set_trace
class TransformerDataset(Dataset):
    """
    Dataset class used for transformer models.
    
    """
    def __init__(self, 
        VmData: torch.tensor,
        ECGData: torch.tensor,
        actTimeData: torch.tensor,
        datInd: list,
        enc_seq_len: int, 
        dec_seq_len: int, 
        target_seq_len: int
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

        self.datInd = datInd

        self.VmData = VmData

        self.ECGData = ECGData

        self.actTime = actTimeData

        self.enc_seq_len = enc_seq_len

        self.dec_seq_len = dec_seq_len

        self.target_seq_len = target_seq_len



    def __len__(self):
        return self.ECGData.shape[0]

    def __getitem__(self, idx):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """
        # Get the first and last element of the i'th tuple in the list self.VmInds

        src_list, trg_list, trg_y_list = [], [], []
        inp_seq = self.ECGData[idx,:,:]
        tar_seq = self.VmData[idx,:,:]
        act_time = self.actTime[idx, :]
        src, trg, trg_y = self.get_src_trg(
            inp_sequence= inp_seq,
            target_sequence = tar_seq,
            enc_seq_len=self.enc_seq_len,
            dec_seq_len=self.dec_seq_len,
            target_seq_len=self.target_seq_len
            )
            

        
        return src, trg, trg_y, act_time
        
    def get_src_trg(
        self,
        inp_sequence: torch.Tensor,
        target_sequence: torch.Tensor,
        enc_seq_len: int, 
        dec_seq_len: int, 
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

        # encoder input
        src = inp_sequence[:enc_seq_len,:]
        trg, trg_y = [], []

        for i in range(len(self.datInd)-1):
            start_trg_idx, end_trg_idx = self.datInd[i][0], self.datInd[i][1]
            start_trgy_idx, end_trgy_idx = self.datInd[i+1][0], self.datInd[i+1][1]
            trg.append(target_sequence[start_trg_idx:end_trg_idx, :])
            trg_y.append(target_sequence[start_trgy_idx:end_trgy_idx, :])
        trg = torch.stack(trg, axis = 0)
        trg_y = torch.stack(trg_y, axis = 0)
        return src, trg, trg_y