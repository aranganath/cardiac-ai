import torch
from torch import nn, Tensor
import math
import torch.nn.functional as F

from pdb import set_trace

class PositionalEncoder(nn.Module):
    """
    The authors of the original transformer paper describe very succinctly what 
    the positional encoding layer does and why it is needed:
    
    "Since our model contains no recurrence and no convolution, in order for the 
    model to make use of the order of the sequence, we must inject some 
    information about the relative or absolute position of the tokens in the 
    sequence." (Vaswani et al, 2017)

    Adapted from: 
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(
        self, 
        dropout: float=0.1, 
        max_seq_len: int=5000, 
        d_model: int=512,
        batch_first: bool=False
        ):

        """
        Parameters:

            dropout: the dropout rate

            max_seq_len: the maximum length of the input sequences

            d_model: The dimension of the output of sub-layers in the model 
                     (Vaswani et al, 2017)
        """

        super().__init__()

        self.d_model = d_model
        
        self.dropout = nn.Dropout(p=dropout)

        self.batch_first = batch_first

        self.x_dim = 1 if batch_first else 0

        # copy pasted from PyTorch tutorial
        position = torch.arange(max_seq_len).unsqueeze(1)
        
        if d_model %2 == 0:
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            
            pe = torch.zeros(max_seq_len, 1, d_model)
            
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            

            pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        else:
            div_term_even = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            
            div_term_odd = torch.exp(torch.arange(1, d_model, 2) * (-math.log(10000.0) / d_model))

            pe = torch.zeros(max_seq_len, 1, d_model)
            
            pe[:, 0, 0::2] = torch.sin(position * div_term_even)
            

            pe[:, 0, 1::2] = torch.cos(position * div_term_odd)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """

        x = x + self.pe[:x.size(self.x_dim)]

        return self.dropout(x)



class TimeSeriesTransformer(nn.Module):

    """
    This class implements a transformer model that can be used for times series
    forecasting. This time series transformer model is based on the paper by
    Wu et al (2020) [1]. The paper will be referred to as "the paper".

    A detailed description of the code can be found in my article here:

    https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e

    In cases where the paper does not specify what value was used for a specific
    configuration/hyperparameter, this class uses the values from Vaswani et al
    (2017) [2] or from PyTorch source code.

    Unlike the paper, this class assumes that input layers, positional encoding 
    layers and linear mapping layers are separate from the encoder and decoder, 
    i.e. the encoder and decoder only do what is depicted as their sub-layers 
    in the paper. For practical purposes, this assumption does not make a 
    difference - it merely means that the linear and positional encoding layers
    are implemented inside the present class and not inside the 
    Encoder() and Decoder() classes.

    [1] Wu, N., Green, B., Ben, X., O'banion, S. (2020). 
    'Deep Transformer Models for Time Series Forecasting: 
    The Influenza Prevalence Case'. 
    arXiv:2001.08317 [cs, stat] [Preprint]. 
    Available at: http://arxiv.org/abs/2001.08317 (Accessed: 9 March 2022).

    [2] Vaswani, A. et al. (2017) 
    'Attention Is All You Need'.
    arXiv:1706.03762 [cs] [Preprint]. 
    Available at: http://arxiv.org/abs/1706.03762 (Accessed: 9 March 2022).

    """

    def __init__(self, 
        input_size: int,
        dec_seq_len: int,
        batch_first: bool,
        out_seq_len: int=58,
        dim_val: int=512,  
        n_encoder_layers: int=4,
        n_decoder_layers: int=4,
        n_heads: int=8,
        dropout_encoder: float=0.2, 
        dropout_decoder: float=0.2,
        dropout_pos_enc: float=0.1,
        dim_feedforward_encoder: int=128,
        dim_feedforward_decoder: int=128,
        num_predicted_features: int=75
        ): 

        """
        Args:

            input_size: int, number of input variables. 1 if univariate.

            dec_seq_len: int, the length of the input sequence fed to the decoder

            dim_val: int, aka d_model. All sub-layers in the model produce 
                     outputs of dimension dim_val

            n_encoder_layers: int, number of stacked encoder layers in the encoder

            n_decoder_layers: int, number of stacked encoder layers in the decoder

            n_heads: int, the number of attention heads (aka parallel attention layers)

            dropout_encoder: float, the dropout rate of the encoder

            dropout_decoder: float, the dropout rate of the decoder

            dropout_pos_enc: float, the dropout rate of the positional encoder

            dim_feedforward_encoder: int, number of neurons in the linear layer 
                                     of the encoder

            dim_feedforward_decoder: int, number of neurons in the linear layer 
                                     of the decoder

            num_predicted_features: int, the number of features you want to predict.
                                    Most of the time, this will be 1 because we're
                                    only forecasting FCR-N prices in DK2, but in
                                    we wanted to also predict FCR-D with the same
                                    model, num_predicted_features should be 2.
        """

        super().__init__()

        self.dec_seq_len = dec_seq_len
        self.num_predicted_features = num_predicted_features

        # Creating the three linear layers needed for the model
        # self.encoder_input_layer = nn.Linear(
        #     in_features=input_size, 
        #     out_features=dim_val 
        #     )

        self.encoder_input_layer = nn.Sequential(
            nn.Conv1d(12, 12, stride=2, kernel_size=3), 
            nn.ReLU(),
            nn.Conv1d(12, 12, kernel_size = 2, stride = 2),
            nn.ReLU(),
            nn.Conv1d(12,12, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Linear(62, dim_val)
        )

        self.decoder_input_layer = nn.Sequential(
            nn.Linear(
                in_features=num_predicted_features,
                out_features=dim_val),
        )
        
        self.linear_mapping = nn.Linear(
            in_features=dim_val, 
            out_features=num_predicted_features
        )

        # Create positional encoder
        self.positional_encoding_layer = PositionalEncoder(
            d_model=dim_val,
            dropout=dropout_pos_enc
        )


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            batch_first=batch_first
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=n_encoder_layers, 
            norm=None
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=n_heads,
            dim_feedforward=dim_feedforward_decoder,
            dropout=dropout_decoder,
            batch_first=batch_first
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=n_decoder_layers, 
            norm=None
        )
        # self.decoder = nn.Sequential(
        #     nn.Conv1d(12, 8, kernel_size=3, stride =2),
        #     nn.ReLU(),
        #     nn.Conv1d(8, 4, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.Conv1d(4, 2, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(2, 4, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(4, 8, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.ConvTranspose1d(8, 12, kernel_size=3, stride=2),
        #     nn.ReLU(),
        #     nn.Linear(71,75)
        # )


        self.activationTime = nn.Sequential(
            nn.Conv1d(12, 60, kernel_size=(5,), stride=(3,)),
            nn.ReLU(),
            nn.Conv1d(60, 45, kernel_size=(5,), stride=(2,)),
            nn.ReLU(),
            nn.Conv1d(45, 15, kernel_size=(2,), stride=(2,)),
            nn.ReLU()
        )

        self.activationToVm = nn.Sequential(         
            nn.ConvTranspose1d(15, 45, kernel_size=(2,), stride=(2,),),
            nn.ReLU(),
            nn.ConvTranspose1d(45, 60, kernel_size=(5,), stride=(2,),),
            nn.ReLU(),
            nn.ConvTranspose1d(60, 12, kernel_size=(5,), stride=(3,),),
            nn.ReLU(),
            nn.Linear(71, 75),
            nn.ReLU()
        )
    

    def forward(self, src: Tensor, tgt: Tensor, src_mask: Tensor=None, 
                tgt_mask: Tensor=None) -> Tensor:
        """
        Returns a tensor of shape:

        [target_sequence_length, batch_size, num_predicted_features]
        
        Args:

            src: the encoder's output sequence. Shape: (S,E) for unbatched input, 
                 (S, N, E) if batch_first=False or (N, S, E) if 
                 batch_first=True, where S is the source sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)

            tgt: the sequence to the decoder. Shape: (T,E) for unbatched input, 
                 (T, N, E)(T,N,E) if batch_first=False or (N, T, E) if 
                 batch_first=True, where T is the target sequence length, 
                 N is the batch size, and E is the number of features (1 if univariate)

            src_mask: the mask for the src sequence to prevent the model from 
                      using data points from the target sequence

            tgt_mask: the mask for the tgt sequence to prevent the model from
                      using data points from the target sequence


        """
        recon = []
        src = self.encoder_input_layer(src) 

        # Pass through the positional encoding layer
        src = self.positional_encoding_layer(src)

        

        src = self.encoder(
            src=src
        )

        # What if the previous info we get is the activation time ?!
        # This would be the input to the decoder
        # The output shape will be [batch_size, 75]
        
        src = self.activationTime(src)

        activation = src.reshape(-1, self.num_predicted_features)
        
        src = self.activationToVm(src)

        # During training 
        if self.train:
            
            for i in range(tgt.shape[1]):

                # Pass decoder input through decoder input layer
                out_tgt = self.decoder_input_layer(tgt[:,i,:,:])

                # Pass through decoder - output shape: [batch_size, target seq len, dim_val]
                out_tgt = self.decoder(tgt=out_tgt, memory=src)

                # Pass through linear mapping
                out_tgt = self.linear_mapping(out_tgt)
            

                recon.append(out_tgt) # shape [batch_size, target seq len]
            
            
            recon = torch.stack(recon, axis = 1)
            
            return recon, activation
        
        else:
            out_tgt = tgt[:,0,:,:]

            for i in range(tgt.shape[1]):

                # Pass decoder input through decoder input layer
                out_tgt = self.decoder_input_layer(out_tgt) 
                
                # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
                out_tgt = self.decoder(tgt=out_tgt, memory=src)

                # Pass through linear mapping
                out_tgt = self.linear_mapping(out_tgt)
            

                recon.append(out_tgt) # shape [batch_size, target seq len]
        
            recon = torch.stack(recon, axis = 1)
            return recon, src
        
        
