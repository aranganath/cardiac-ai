from torch.utils.data import DataLoader

from utils import *
from matplotlib import pyplot
import os
# from pdb import set_trace
import seaborn as sns
import matplotlib.pyplot as plt
from models import SqueezeNet
from dataset import CNN1dDataset
from trainable_model import TrainableModel

seed_value = 12345
torch.manual_seed(seed_value)  # For CPU
torch.cuda.manual_seed(seed_value)  # For GPU

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.set_default_device(device)
print(device)
generator = torch.Generator(device=device)
torch.set_default_dtype(torch.float64)

# Load data
# Reading file pairs -- Mikel's way
data_paths=[]
for dir in os.listdir('./intracardiac_dataset/'):
    if "data_hearts_dd_0p2" in dir:
        data_paths.append('./intracardiac_dataset/'+ dir)   
        
# datapaths_train = './intracardiac_dataset/data_hearts_dd_0p2'
file_pairs = read_data_dirs(data_paths)
data_scaling_ecg = 'normalized_unit'
data_scaling_vm = 'normalization'
num_timesteps = 500
all_channels = True
input_dim = 12
output_dim = 75
outLead = [i for i in range(output_dim)]
train_len = 12000
val_len = 4000

# Load training and validation sets normalized
TrainData = Ecg2VmDataset(file_pairs[:train_len], outLead, num_timesteps, 
              scaling_ecg= data_scaling_ecg, scaling_vm= data_scaling_vm, 
              noise_ecg= 'none', noise_vm= 'none')
ValData = Ecg2VmDataset(file_pairs[train_len:train_len+val_len], outLead, num_timesteps, 
              scaling_ecg= data_scaling_ecg, scaling_vm= data_scaling_vm, 
              noise_ecg= 'none', noise_vm= 'none')

# path = './intracardiac_dataset/'
# VmTrainData, pECGTrainData, VmDataTest, pECGTestData, actTimeTrain, actTimeTest  = fileReader(path, 16000, 0.8)
# print('Data loading from files - complete')

# VmTrainData = (VmTrainData - torch.min(VmTrainData))/(torch.max(VmTrainData) - torch.min(VmTrainData))
# pECGTrainData = (pECGTrainData - torch.min(pECGTrainData))/(torch.max(pECGTrainData) - torch.min(pECGTrainData))

# VmDataTest = (VmDataTest - torch.min(VmDataTest))/(torch.max(VmDataTest) - torch.min(VmDataTest))

# pECGTestData = (pECGTestData - torch.min(pECGTestData))/(torch.max(pECGTestData) - torch.min(pECGTestData))
# print('Normalization - complete!')

# 80 -10 -10 split 
# TrainData = CNN1dDataset(VmTrainData, pECGTrainData)
# ValData = CNN1dDataset(VmDataTest, pECGTestData)
#TestData = CNN1dDataset(VmDataTest[data_split:], pECGTestData[data_split:])

model = SqueezeNet(version='1_1')
model.to(device)

# Define learning parameters -- same as Mikel's github
learning_rate = 1e-3
gamma_scheduler = 1.0
step_size_scheduler = 100
batch_size = 32
num_epochs = 1400
grad_clippling = False
dropout = 0.0
# loss_norm = MSE
load_model = False
model_path = '/home/jornelasmunoz/cardiac-ai/cnnModel_v1.training.epoch.36.pth'#./mymodel.training.epoch.1400.pth
if load_model:
    model = torch.load(model_path)
    model.to(device)
    print(f"Loading model from {model_path}")
    
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size_scheduler,gamma=gamma_scheduler)
outputHandler = '../cardiac-ai/trainingStats'

# Define data loader
dataloader = DataLoader(TrainData, batch_size)
val_loader = DataLoader(ValData, batch_size)

train_model = TrainableModel(criterion, optimizer, scheduler, outputHandler, device=device, progressbar = True)
train_model.learn(model, dataloader, val_loader, num_epochs, grad_clippling=False, checkpointRate = None, name = "cnnModel_v1_1")
