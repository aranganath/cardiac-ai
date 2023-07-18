import os
import torch
from torch.utils.data import Dataset
from typing import Tuple

class TransformerDataset(Dataset):
    """
    Dataset class used for transformer models.
    
    """
    def __init__(self, 
        VmData: torch.tensor,
        ECGData: torch.tensor,
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

        self.enc_seq_len = enc_seq_len

        self.dec_seq_len = dec_seq_len

        self.target_seq_len = target_seq_len



    def __len__(self):
        
        return len(self.datInd)

    def __getitem__(self, index):
        """
        Returns a tuple with 3 elements:
        1) src (the encoder input)
        2) trg (the decoder input)
        3) trg_y (the target)
        """
        # Get the first and last element of the i'th tuple in the list self.VmInds

        src_list, trg_list, trg_y_list = [], [], []

        for i in range(len(self.datInd)-1):
            start_idx, end_idx = self.datInd[i][0], self.datInd[i][1]

            output_sequence = self.VmData[index,start_idx:end_idx,:]

            input_sequence = self.ECGData[index, start_idx:end_idx, :]


            src, trg, trg_y = self.get_src_trg(
                inp_sequence= input_sequence,
                target_sequence = output_sequence,
                enc_seq_len=self.enc_seq_len,
                dec_seq_len=self.dec_seq_len,
                target_seq_len=self.target_seq_len
                )
            
            src_list.append(src)
            trg_list.append(trg)
            trg_y_list.append(trg_y)
        
        
        return torch.stack(src_list, axis =1), torch.stack(trg_list, axis = 1), torch.stack(trg_y_list, axis =1)
        
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

        assert inp_sequence.shape[0] == enc_seq_len + target_seq_len, "Sequence length {} does not equal encoder length {} + target length {}".format(sequence.shape[2],enc_seq_len, target_seq_len)
        
        # encoder input
        src = inp_sequence[:enc_seq_len] 
        
        # decoder input. As per the paper, it must have the same dimension as the 
        # target sequence, and it must contain the last value of src, and all
        # values of trg_y except the last (i.e. it must be shifted right by 1)
        trg = target_sequence[enc_seq_len-1:target_sequence.shape[0]-1]
        assert trg.shape[0] == target_seq_len, "Length of trg {trg} does not match target sequence length {target_seq_len}".format(trg=trg.shape[0], target_seq_len=target_seq_len)

        # The target sequence against which the model output will be compared to compute loss
        trg_y = target_sequence[enc_seq_len: target_sequence.shape[0]]

        assert trg_y.shape[0] == target_seq_len, "Length of trg_y does not match target sequence length"

        return src, trg, trg_y # change size from [batch_size, target_seq_len, num_features] to [batch_size, target_seq_len] 