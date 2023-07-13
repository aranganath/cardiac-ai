from models import TimeSeriesTransformer
from dataset import TransformerDataset
from torch.utils.data import DataLoader
from utils import *
import os


torch.set_default_dtype(torch.float64)
# Okay, we kind of have a way of loading the data
# We need to collect the data and feed it to the transformer model
# Now how do we that ?


# Right, so let's get the data


# Torch Tensor data !
path = './Datasets/intracardiac_dataset/'
VmData, pECGData = fileReader(path)
print('Data loading from files - complete')
## Model parameters
dim_val = 512 # This can be any value divisible by n_heads. 512 is used in the original transformer paper.
n_heads = 8 # The number of attention heads (aka parallel attention layers). dim_val must be divisible by this number
n_decoder_layers = 4 # Number of times the decoder layer is stacked in the decoder
n_encoder_layers = 4 # Number of times the encoder layer is stacked in the encoder
input_size = 75 # The number of input variables. 1 if univariate forecasting.
dec_seq_len = 100 # length of input given to decoder. Can have any integer value.
enc_seq_len = 200 # length of input given to encoder. Can have any integer value.
output_sequence_length = 200 # Length of the target sequence, i.e. how many time steps should your forecast cover
max_seq_len = enc_seq_len # What's the longest sequence the model will encounter? Used to make the positional encoder
batch_size = 128
batch_first= False

# Get the indices of the sequences
# The idea is: start - stop, where stop - start is window_size
# This means, each tuple in VmInd and pECGInd is 50 steps
VmInd, pECGInd = get_indices_entire_sequence(VmData = VmData, 
                                            ECGData = pECGData, 
                                            window_size= enc_seq_len + output_sequence_length, 
                                            step_size = 1)

# Now let's collect the training data in the Transformer Dataset class

training_data = TransformerDataset(VmData = VmData, 
                                    ECGData = pECGData,
                                    VmIndices= VmInd,
                                    ECGInd = pECGInd,
                                    enc_seq_len = enc_seq_len,
                                    dec_seq_len = dec_seq_len,
                                    target_seq_len = output_sequence_length)


trainLoader = DataLoader(training_data, batch_size)


model = TimeSeriesTransformer(
    dim_val=dim_val,
    batch_first=batch_first,
    input_size=input_size, 
    dec_seq_len=dec_seq_len,
    out_seq_len=output_sequence_length, 
    n_decoder_layers=n_decoder_layers,
    n_encoder_layers=n_encoder_layers,
    n_heads=n_heads,
    num_predicted_features=input_size)

# Define the MSE loss
criterion = torch.nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)


# Permute from shape [batch size, seq len, num features] to [seq len, batch size, num features]
EPOCHS = 10
src, trg, trg_y = next(iter(trainLoader))
src_mask = generate_square_subsequent_mask(
            dim1=output_sequence_length,
            dim2=enc_seq_len
        )
tgt_mask = generate_square_subsequent_mask(
    dim1=output_sequence_length,
    dim2=output_sequence_length
)
src = src.permute(1, 0, 2)
trg = trg.permute(1, 0, 2)
prediction = model(
src=src,
tgt=trg,
src_mask=src_mask,
tgt_mask=tgt_mask
)

loss = criterion(trg_y , prediction.permute(1,0,2))

# Iterate over all epochs
for epoch in tqdm(range(EPOCHS), desc = 'Training ', unit='epochs'):

    for src, trg, trg_y in trainLoader:
        optimizer.zero_grad()
        src_mask = generate_square_subsequent_mask(
            dim1=output_sequence_length,
            dim2=enc_seq_len
        )
        tgt_mask = generate_square_subsequent_mask(
            dim1=output_sequence_length,
            dim2=output_sequence_length
        )
        src = src.permute(1, 0, 2)
        trg = trg.permute(1, 0, 2)
        prediction = model(
            src=src,
            tgt=trg,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        loss = criterion(trg_y, prediction.permute(1,0,2))
        print('Loss: ',loss.item())
        loss.backward()
        optimizer.step()