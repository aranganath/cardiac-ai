import torch
from torch import nn, Tensor
import math
import torch.nn.functional as F
from base import BaseModel
from fire import Fire
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
        max_seq_len: int=50, 
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
        position = torch.arange(max_seq_len)
        
        if d_model %2 == 0:
            div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            
            pe = torch.zeros(1, d_model, max_seq_len)
            
            pe[:, 0, 0::2] = torch.sin(position * div_term)
            

            pe[:, 0, 1::2] = torch.cos(position * div_term)
        
        else:
            div_term_even = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
            
            div_term_odd = torch.exp(torch.arange(1, d_model, 2) * (-math.log(10000.0) / d_model))

            pe = torch.zeros(1, d_model, max_seq_len)
            
            pe[0, 0::2, :] = torch.sin(div_term_even.unsqueeze(1)*position)

            pe[0, 1::2, :] = torch.cos(div_term_odd.unsqueeze(1)*position)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, enc_seq_len, dim_val] or 
               [enc_seq_len, batch_size, dim_val]
        """
        
        x = x + self.pe[:x.size(self.x_dim)]

        return self.dropout(x)



class TimeSeriesTransformer(BaseModel):

    """
        This class implements a transformer model that can be used for times series
        forecasting. This time series transformer model is based on the paper by
        Wu et al (2020) [1]. The paper will be referred to as "the paper".

        A detailed description of the code can be found in my article here:

        https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
    """

    def __init__(
            self,
            in_channels:int, 
            loss_fn,
            batch_first: bool,
            dim_val: int=512,  
            enc_layers: int=4,
            dec_layers: int=4,
            heads: int=8,
            enc_drop: float=0.2, 
            dec_drop: float=0.2,
            dropout_pos_enc: float=0.1,
            enc_feed: int=256,
            dec_feed: int=256,
            out_channels: int=75,
            steps: int=100
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

        super(TimeSeriesTransformer, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            loss_fn=loss_fn
        )
        kernel_size = 3
        self.encoder_input_layer = nn.Sequential(
                nn.Conv1d(12, 64, kernel_size = kernel_size, stride=1, padding = 1 + math.ceil((3-3)/2) ),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size = kernel_size, stride=1, padding = 1 + math.ceil((3-3)/2), ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding = 1 + math.floor((kernel_size-3)/2), ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding = 1 + math.floor((kernel_size-3)/2), ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
                nn.Dropout(p = 0.5),
                nn.Conv1d(512, out_channels=out_channels, kernel_size=1, padding = 0)

        )

        self.positional_encoding_layer = PositionalEncoder(
            d_model=self.out_channels,
            dropout=dropout_pos_enc,
            batch_first=batch_first,
            max_seq_len=steps
        )


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=heads,
            dim_feedforward=enc_feed,
            dropout=enc_drop,
            batch_first=batch_first
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=enc_layers, 
            enable_nested_tensor=False,
            norm=None
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=heads,
            dim_feedforward=dec_feed,
            dropout=dec_drop,
            batch_first=batch_first
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=dec_layers, 
            norm=None
        )
    

    def _forward(self, inputs:dict) -> Tensor:
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
        
        src = inputs["x"]
        tgt = inputs["y"]
        
        src = self.encoder_input_layer(src) 

        # Pass through the positional encoding layer
        src = self.positional_encoding_layer(src)        
        
        src = self.encoder(
            src=src.permute(0,2,1)
        )

        # What if the previous info we get is the activation time ?!
        # This would be the input to the decoder
        # The output shape will be [batch_size, 75]
        # Generate the first k timesteps (This should be your window)
        recon.append(src)
        
        # During training 
        if self.train:
            for i in range(tgt.shape[1]-1):

                # Pass through decoder - output shape: [batch_size, target seq len, dim_val]
                out_tgt = self.decoder(tgt=tgt[:,i,:,:], memory=src)

                recon.append(out_tgt) # shape [batch_size, target seq len]
            
            recon = torch.stack(recon, axis = 1)
            
            return recon
        
        else:

            for i in range(tgt.shape[1]):

                
                # Pass throguh decoder - output shape: [batch_size, target seq len, dim_val]
                out_tgt = self.decoder(tgt=recon[-1], memory=src)

                # Pass through linear mapping
                recon.append(out_tgt) # shape [batch_size, target seq len]
        
            recon = torch.stack(recon, axis = 1)
            return recon
        
        
class SelfTimeSeriosTransformer(BaseModel):

    """
        This class implements a transformer model that can be used for times series
        forecasting. This time series transformer model is based on the paper by
        Wu et al (2020) [1]. The paper will be referred to as "the paper".

        A detailed description of the code can be found in my article here:

        https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e
    """

    def __init__(
            self,
            in_channels:int, 
            loss_fn,
            batch_first: bool,
            dim_val: int=512,  
            enc_layers: int=4,
            dec_layers: int=4,
            heads: int=8,
            enc_drop: float=0.2, 
            dec_drop: float=0.2,
            dropout_pos_enc: float=0.1,
            enc_feed: int=256,
            dec_feed: int=256,
            out_channels: int=75,
            steps: int=100
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

        super(SelfTimeSeriosTransformer, self).__init__(
            in_channels=in_channels, 
            out_channels=out_channels, 
            loss_fn=loss_fn
        )
        kernel_size = 3
        self.encoder_input_layer = nn.Sequential(
                nn.Conv1d(12, 64, kernel_size = kernel_size, stride=1, padding = 1 + math.ceil((3-3)/2) ),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size = kernel_size, stride=1, padding = 1 + math.ceil((3-3)/2), ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding = 1 + math.floor((kernel_size-3)/2), ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool1d(kernel_size=kernel_size, stride=1, padding = 1 + math.floor((kernel_size-3)/2), ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
                nn.Dropout(p = 0.5),
                nn.Conv1d(512, out_channels=out_channels, kernel_size=1, padding = 0)

        )

        self.positional_encoding_layer = PositionalEncoder(
            d_model=self.out_channels,
            dropout=dropout_pos_enc,
            batch_first=batch_first,
            max_seq_len=steps
        )


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_val, 
            nhead=heads,
            dim_feedforward=enc_feed,
            dropout=enc_drop,
            batch_first=batch_first
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=enc_layers, 
            enable_nested_tensor=False,
            norm=None
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim_val,
            nhead=heads,
            dim_feedforward=dec_feed,
            dropout=dec_drop,
            batch_first=batch_first
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer,
            num_layers=dec_layers, 
            norm=None
        )
    

    def _forward(self, inputs:dict) -> Tensor:
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
        
        src = inputs["x"]
        tgt = inputs["y"]
        
        src = self.encoder_input_layer(src) 

        # Pass through the positional encoding layer
        src = self.positional_encoding_layer(src)        
        
        src = self.encoder(
            src=src.permute(0,2,1)
        )

        # What if the previous info we get is the activation time ?!
        # This would be the input to the decoder
        # The output shape will be [batch_size, 75]
        # Generate the first k timesteps (This should be your window)
        recon.append(src)
        
        # During training 
        for i in range(tgt.shape[1]-1):

            # Pass through decoder - output shape: [batch_size, target seq len, dim_val]
            out_tgt = self.decoder(tgt=recon[-1], memory=src)

            recon.append(out_tgt) # shape [batch_size, target seq len]
        
        recon = torch.stack(recon, axis = 1)
        
        return recon        