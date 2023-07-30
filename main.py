# %%
from torch.utils.data import DataLoader
from utils import *
from matplotlib import pyplot
import os
from pdb import set_trace
from dataset import TransformerDataset

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
print(device)

# %%
torch.set_default_dtype(torch.float64)
# Okay, we kind of have a way of loading the data
# We need to collect the data and feed it to the transformer model
# Now how do we that ?


# Torch Tensor data !
path = './Datasets/intracardiac_dataset/'
VmTrainData, pECGTrainData, VmDataTest, pECGTestData  = fileReader(path, 100)
print('Data loading from files - complete')

VmTrainData = (VmTrainData - torch.min(VmTrainData))/(torch.max(VmTrainData) - torch.min(VmTrainData))
pECGTrainData = (pECGTrainData - torch.min(pECGTrainData))/(torch.max(pECGTrainData) - torch.min(pECGTrainData))

VmDataTest = (VmDataTest - torch.min(VmDataTest))/(torch.max(VmDataTest) - torch.min(VmDataTest))

pECGTestData = (pECGTestData - torch.min(pECGTestData))/(torch.max(pECGTestData) - torch.min(pECGTestData))
print('Normalization - complete!')

# %%
dim_val = 320
n_heads = 32
n_decoder_layers = 2
n_encoder_layers = 2
input_size = 12
dec_seq_len = 498
enc_seq_len = 500
output_sequence_length = 75
max_seq_len = enc_seq_len
train_batch_size = 8
test_batch_size = 10
batch_first= False
output_size = 75
window_size = 75

# %%

# Get the indices of the sequences
# The idea is: start - stop, where stop - start is window_size
# This means, each tuple in VmInd and pECGInd is 50 steps
datInd = get_indices_entire_sequence(VmData = VmTrainData, 
                                            ECGData = pECGTrainData, 
                                            window_size= window_size, 
                                            step_size = window_size)

# Now let's collect the training data in the Transformer Dataset class
TrainData = TransformerDataset(VmData = VmTrainData,
                                    datInd=datInd,
                                    ECGData = pECGTrainData,
                                    enc_seq_len = enc_seq_len,
                                    dec_seq_len = dec_seq_len,
                                    target_seq_len = output_sequence_length
                                )


TrainData = DataLoader(TrainData, train_batch_size)

datInd = get_indices_entire_sequence(VmData = VmDataTest, 
                                            ECGData = pECGTestData, 
                                            window_size= window_size, 
                                            step_size = window_size)


# Now, let's load the test data
TestData = TransformerDataset(VmData = VmDataTest, 
                                    ECGData = pECGTestData,
                                    datInd=datInd,
                                    enc_seq_len = enc_seq_len,
                                    dec_seq_len = dec_seq_len,
                                    target_seq_len = output_sequence_length
                                )

TestData = DataLoader(TestData, test_batch_size)

# %%
from models import TimeSeriesTransformer
model = TimeSeriesTransformer(
    dim_val=dim_val,
    batch_first=batch_first,
    input_size=input_size, 
    dec_seq_len=dec_seq_len,
    out_seq_len=output_sequence_length, 
    n_decoder_layers=n_decoder_layers,
    n_encoder_layers=n_encoder_layers,
    n_heads=n_heads,
    num_predicted_features=output_size
)

# model = torch.nn.DataParallel(model)

# Define the MSE loss
criterion = torch.nn.HuberLoss(delta=5.0)

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.1)

# %%
EPOCHS= 80000
train_losses = []
src_mask = generate_square_subsequent_mask(
                dim1=output_sequence_length,
                dim2=enc_seq_len
            )
tgt_mask = generate_square_subsequent_mask(
    dim1=output_sequence_length,
    dim2=output_sequence_length
)

gap_length = 1
pbar = tqdm(range(EPOCHS), desc='Training')

for epoch in pbar:
    string = ''
    for src, trg, trg_y in TrainData:
        optimizer.zero_grad()
        prediction = model(
            src=src.permute(1,0,2).to(device),
            tgt=trg.permute(0,2,1,3).permute(1,0,2,3).to(device),
            src_mask=src_mask,
            tgt_mask=tgt_mask
            )
        
        
        loss = criterion(prediction.view_as(trg).to(device), trg_y.to(device))
        loss.backward()
        optimizer.step()
        if (epoch+1) % gap_length == 0:
            model.train = True
            with torch.no_grad():
                src, trg, trg_y = next(iter(TestData))
                prediction = model(
                        src=src.permute(1,0,2).to(device),
                        tgt=trg.permute(0,2,1,3).permute(1,0,2,3).to(device),
                        src_mask=src_mask,
                        tgt_mask=tgt_mask
                    )
                row = 7
                column = 10
                

                prediction = prediction.view_as(trg_y).reshape(-1,trg_y.shape[1]*trg_y.shape[2] , 75).detach().cpu()
                trg_y = trg_y.reshape(-1, trg_y.shape[1]*trg_y.shape[2], 75).detach().cpu()
                pyplot.figure(figsize=(18, 9))
                for count, i in enumerate(range(prediction.shape[2])):
                    pyplot.subplot(8, 10, count + 1)
                    pyplot.plot(prediction[0,:,i])
                    pyplot.plot(trg_y[0,:,i])
                    pyplot.title(f'i = {i}')
                    pyplot.grid(visible=True, which='major', color='#666666', linestyle='-')
                    pyplot.minorticks_on()
                    pyplot.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
                pyplot.tight_layout()
                if not os.path.isdir('graphs'):
                    os.mkdir('graphs')
                pyplot.savefig('./graphs/model_'+str(dim_val) +'_encoder_'+ str(n_encoder_layers)+'_decoder_'+str(n_decoder_layers)+'_epochs_'+str(epoch)+'_window_size_'+str(window_size)+'.png')

                PARENT_PATH = 'model_weights'
                if not os.path.isdir(PARENT_PATH):
                    os.mkdir(PARENT_PATH)
                
                PATH = './model_weights/model_'+str(dim_val) +'_encoder_'+ str(n_encoder_layers)+'_decoder_'+str(n_decoder_layers)+'_epochs_'+str(EPOCHS)+'.pth'
                sring = 'Saved to path: '+PATH
                torch.save(model.state_dict(), PATH)

        
    
    pbar.set_description('Training   Loss: '+str(loss.item())+string)
    train_losses.append(loss.item())