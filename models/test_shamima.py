import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import r2_score
import glob, re, os
import numpy as np
from typing import List
import matplotlib.pyplot as plt
from cardiac_ml_tools import read_data_dirs, get_standard_leads, get_activation_time
from scipy import stats
import pandas as pd

data_dirs = []
regex = r'data_hearts_dd_0p2*'
DIR = '/usr/workspace/hossain4/cardiac_ai/intracardiac_dataset/'

for x in os.listdir(DIR):
    if re.match(regex,x):
        data_dirs.append(DIR+x)

file_pairs = read_data_dirs(data_dirs)

batch_size = 117
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)
    
    def __getitem__(self,idx):
        feature = self.features[idx]
        label = self.labels[idx]
        return feature, label
    
feature, label = [], []

for case in range(len(file_pairs)):
    pECGData = np.load(file_pairs[case][0])
    pECGData = get_standard_leads(pECGData)

    VmData = np.load(file_pairs[case][1])

    label.append(torch.tensor(VmData).to(device))
    feature.append(torch.tensor(pECGData, dtype=torch.float32))

feature = torch.stack(feature, dim = 0)

train_feature, train_label = feature[:12894], label[:12894]
test_feature, test_label = feature[16000:], label[16000:]

lead_data_min = torch.min(torch.min(train_feature, dim = 1).values, dim = 0).values
lead_data_min = lead_data_min.reshape(1,1,-1)

lead_data_max = torch.max(torch.max(train_feature, dim = 1).values, dim = 0).values
lead_data_max = lead_data_max.reshape(1,1,-1)

train_feature = (train_feature - lead_data_min) / (lead_data_min - lead_data_max)
test_feature = (test_feature - lead_data_min) / (lead_data_min - lead_data_max)

act_data_min = torch.min(torch.min(torch.stack(train_label, dim = 0), dim = 1).values, dim = 0).values
act_data_min = act_data_min.reshape(1,1,-1).to(device)

act_data_max = torch.max(torch.max(torch.stack(train_label, dim = 0), dim = 1).values, dim = 0).values
act_data_max = act_data_max.reshape(1,1,-1).to(device)

test_label = [(l - act_data_min)/(act_data_max - act_data_min) for l in test_label]

cid = CustomDataset(test_feature, test_label)

valid_data = torch.utils.data.DataLoader(
    cid,
    batch_size = batch_size,
    shuffle = True,
    drop_last = True,
    num_workers = 0,
)

class myModel(BaseModel):
    def __init__(self, output_dim):
        super(myModel, self).__init__()
        self.model = nn.Sequential(
            nn.Conv1d(12,1024, kernel_size = 3),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=3, stride = 1),
            nn.Conv1d(1024,512, kernel_size = 3),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=3, stride = 1),
            nn.Conv1d(512,256, kernel_size = 3),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=3, stride = 1),
            nn.Conv1d(256,128, kernel_size = 3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=3, stride = 1),
            nn.Conv1d(128,75, kernel_size = 3),
            nn.BatchNorm1d(75),
            nn.ReLU(inplace = True),
            nn.MaxPool1d(kernel_size=5, stride = 1),
        )

        self.avg_pool = nn.AdaptiveAvgPool1d(500)
        self.fc = nn.Linear(500, output_dim)
        self.sigmoid = nn.Sigmoid()

    
    def _forward(self, x):
        x = self.model(x)
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
model = myModel(output_dim= 500)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)
scheduler = ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 25, verbose = True, min_lr = 1e-6)

model = model.to(device)

path = "/usr/workspace/hossain4/cardiac_ai/cardiac-ai/checkpoint_4.tar"
checkpoint = torch.load(path)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

valid_err = 0
valid_avg_error = torch.tensor([]).to(device)
valid_mse_error = torch.tensor([]).to(device)
true_labels = []
predicted_labels = []

model.eval()
with torch.no_grad():
    for feature,label in valid_data:
        feature = feature.to(torch.float32).to(device)
        y_pred = model(feature.permute(0,2,1))
        label = label.float().to(device)

        label = label.squeeze(1)
        y_pred = y_pred.permute(0,2,1)

        #y_pred = y_pred*(act_data_max - act_data_min) + act_data_min
        #label = label*(act_data_max - act_data_min) + act_data_min

        valid_avg_error = torch.cat((torch.reshape((torch.sum(abs(label-y_pred))/(batch_size*75)),(-1,)),valid_avg_error), dim = 0).to(device)
        valid_mse_error = torch.cat((torch.reshape((torch.sum(abs(label-y_pred)**2)/(batch_size*75)),(-1,)),valid_mse_error), dim = 0).to(device)
        true_labels.extend(label.squeeze(1).cpu().numpy().tolist())
        predicted_labels.extend(y_pred.squeeze(2).detach().cpu().numpy().tolist())

        
        print(f'Valid Avg Er: {float(torch.sum(valid_avg_error)/len(valid_avg_error))} Valid RMSE : {float(torch.sqrt(torch.sum(valid_mse_error)/len(valid_mse_error)))}')
        true_labels = np.array(true_labels).flatten()
        predicted_labels = np.array(predicted_labels).flatten()

        predicted_labels_pd = pd.DataFrame(predicted_labels, columns = ['Prediction'])
        predicted_labels_pd['Actual'] = true_labels
        predicted_labels_pd.to_csv('/usr/workspace/hossain4/cardiac_ai/test_4.csv')
        r2 = r2_score(true_labels, predicted_labels)
        print(f'R2 score: {r2:.4f}')
        rmse = np.sqrt(np.mean((true_labels - predicted_labels)**2))
        print('rmse',  rmse)
        pearson = stats.pearsonr(true_labels, predicted_labels)
        spearman = stats.spearmanr(true_labels, predicted_labels, alternative = 'greater')
        print(f'R2 score: {r2:.4f}')
        print('pearson', pearson)
        print('spearman', spearman)
