# %%
import torch
from torch.utils.data import DataLoader
from util import *
from matplotlib import pyplot
import os
from pdb import set_trace
from dataset import TransformerDataset
from models1 import TimeSeriesTransformer

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

# Now, also get the activation times
path = './Datasets/intracardiac_dataset/'
train_test_ratio = 0.8
VmTrainData, pECGTrainData, VmDataTest, pECGTestData, actTimeTrain, actTimeTest  = fileReader(path, 20, train_test_ratio)
print('Data loading from files - complete')

VmTrainData = (VmTrainData - torch.min(VmTrainData))/(torch.max(VmTrainData)-torch.min(VmTrainData))
pECGTrainData = (pECGTrainData - torch.min(pECGTrainData))/(torch.max(pECGTrainData) - torch.min(pECGTrainData))

VmDataTest = (VmDataTest - torch.min(VmDataTest))/(torch.max(VmDataTest) - torch.min(VmDataTest))

pECGTestData = (pECGTestData - torch.min(pECGTestData))/(torch.max(pECGTestData) - torch.min(pECGTestData))
print('Normalization - complete!')

# %%
dim_val = 75
n_heads = 75
n_decoder_layers = 1
n_encoder_layers = 1
input_size = 12
dec_seq_len = 498
enc_seq_len = 500

max_seq_len = enc_seq_len
train_batch_size = 20
test_batch_size = 1
batch_first= True
output_size = 75
window_size = 75
stride = window_size
output_sequence_length = window_size

# %%

# Get the indices of the sequences
# The idea is: start - stop, where stop - start is window_size
# This means, each tuple in VmInd and pECGInd is 50 steps
datInd = get_indices_entire_sequence(VmData = VmTrainData, 
                                    ECGData = pECGTrainData,
                                    window_size= window_size, 
                                    step_size = stride
                                )

# Now let's collect the training data in the Transformer Dataset class
TrainData = TransformerDataset(VmData = VmTrainData,
                                    datInd=datInd,
                                    ECGData = pECGTrainData,
                                    actTimeData=actTimeTrain,
                                    enc_seq_len = enc_seq_len,
                                    dec_seq_len = dec_seq_len,
                                    target_seq_len = output_sequence_length
                                )


TrainData = DataLoader(TrainData, batch_size=train_batch_size, shuffle=False, generator=torch.Generator(device='cuda'))
print(TrainData.batch_size)

datInd = get_indices_entire_sequence(VmData = VmDataTest, 
                                            ECGData = pECGTestData, 
                                            window_size= window_size, 
                                            step_size = stride)


# Now, let's load the test data
TestData = TransformerDataset(
                            VmData = VmDataTest, 
                            ECGData = pECGTestData,
                            actTimeData=actTimeTest,
                            datInd=datInd,
                            enc_seq_len = enc_seq_len,
                            dec_seq_len = dec_seq_len,
                            target_seq_len = output_sequence_length
            )

TestData = DataLoader(TestData, test_batch_size, shuffle=False, generator=torch.Generator(device='cuda'))

# %%

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

print('Total Model parameters:', sum([parameter.numel() for parameter in model.parameters()]))


# Define the MSE loss
criterion = torch.nn.HuberLoss(delta=1)

# Define cross-entropy loss for the activation times
criterion2 = torch.nn.MSELoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.999)

EPOCHS= 80000
train_losses = []
src_mask = generate_square_subsequent_mask(
                dim1=output_sequence_length,
                dim2=6
            )
tgt_mask = generate_square_subsequent_mask(
    dim1=output_sequence_length,
    dim2=6
)

train_interval = 100
model_interval = 10000
pbar = tqdm(range(EPOCHS), desc='Training')

for epoch in pbar:
    PATH = ''
    running_loss = 0
    for i, (src, trg, trg_y, act_time) in enumerate(TrainData):
        optimizer.zero_grad()
        recon, activation = model(
            src=src.permute(0,2,1)[:,:,:50].to(device),
            tgt=trg.to(device),
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
        y = torch.cat([trg[:,0,:,:].unsqueeze(1) , trg_y], axis = 1)
        loss = criterion(recon.to(device), y.to(device)) + criterion2(activation.to(device), act_time.type(torch.float64).to(device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        running_loss += loss.item()
        if (epoch+1) % train_interval == 0:
            model.train = False
            with torch.no_grad():
                src, trg, trg_y, act_time = next(iter(TestData))
                recon, activation = model(
                    src=src.permute(0,2,1)[:,:,:50].to(device),
                    tgt=trg.to(device),
                    src_mask=src_mask,
                    tgt_mask=tgt_mask
                )
                y = torch.cat([trg[:,0,:,:].unsqueeze(1) , trg_y], axis = 1)
                row = 7
                column = 10

                recon = recon.reshape(recon.shape[1]*recon.shape[2] , 75).detach().cpu()
                y = y.reshape(y.shape[1]*y.shape[2], 75).detach().cpu()
                pyplot.figure(figsize=(18, 9))
                for count, i in enumerate(range(recon.shape[1])):
                    pyplot.subplot(8, 10, count + 1)
                    pyplot.plot(recon[:,i])
                    pyplot.plot(y[:,i])
                    pyplot.title(f'i = {i}')
                    pyplot.grid(visible=True, which='major', color='#666666', linestyle='-')
                    pyplot.minorticks_on()
                    pyplot.grid(visible=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
                pyplot.tight_layout()
                pyplot.savefig('./graphs/VmRec-'+str(dim_val) +'-encoder-'+ str(n_encoder_layers)+'-decoder-'+str(n_decoder_layers)+'-epochs-'+str(epoch+1)+'-window_size-'+str(window_size)+'.png')
                pyplot.close()
            if (epoch + 1) % model_interval == 0:
                PARENT_PATH = 'model_weights'
                if not os.path.isdir(PARENT_PATH):
                    os.mkdir(PARENT_PATH)
                
                PATH = './model_weights/model-'+str(dim_val) +'-encoder-'+ str(n_encoder_layers)+'-decoder-'+str(n_decoder_layers)+'-epochs-'+str(epoch+1)+'.pth'
                torch.save(model.state_dict(), PATH)
            model.train = True

        
    pbar.set_description('Training   Loss: '+'{:.5f}'.format(running_loss/(i+1))+ ' Saved to :'+PATH)
    train_losses.append(running_loss/(i+1))

# Plotting the training graph
pyplot.figure()
pyplot.plot(train_losses)
if not os.path.isdir('train_graphs'):
    os.mkdir('train_graphs')
TRAINPATH = './train_graphs/VmRec-'+str(dim_val) +'-encoder-'+ str(n_encoder_layers)+'-decoder-'+str(n_decoder_layers)+'-epochs-'+str(epoch)+'-window_size-'+str(window_size)+'.png'
pyplot.savefig(TRAINPATH)
print('Training graph saved to '+ TRAINPATH)