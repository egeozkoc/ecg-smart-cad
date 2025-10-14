
import pandas as pd
import numpy as np
from scipy import signal
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from numpy.random import seed
seed(42)
import random
import warnings
from torch import sigmoid


warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#ECG Data
def get_data(path, selected_outcome):
    
    train_df = pd.read_csv('train_data.csv')
    val_df = pd.read_csv('val_data.csv')
    test_df = pd.read_csv('test_data.csv')
    train_outcomes = train_df[selected_outcome].to_numpy()
    val_outcomes = val_df[selected_outcome].to_numpy()
    test_outcomes = test_df[selected_outcome].to_numpy()
    train_ids = train_df['id'].to_list()
    val_ids = val_df['id'].to_list()
    test_ids = test_df['id'].to_list()

    # get train data
    train_data = []
    for id in train_ids:
        ecg = np.load(path + id + '.npy', allow_pickle=True).item()
        ecg = ecg['waveforms']['ecg_median']
        ecg = ecg[:,150:-50]
        ecg = signal.resample(ecg, 200, axis=1)
        max_val = np.max(np.abs(ecg), axis=1)
        if np.sum(max_val) > 0:
            ecg = ecg / max_val[:, None]
        train_data.append(ecg)
    train_data = np.array(train_data)

    val_data = []
    for id in val_ids:
        ecg = np.load(path + id + '.npy', allow_pickle=True).item()
        ecg = ecg['waveforms']['ecg_median']
        ecg = ecg[:,150:-50]
        ecg = signal.resample(ecg, 200, axis=1)
        max_val = np.max(np.abs(ecg), axis=1)
        if np.sum(max_val) > 0:
            ecg = ecg / max_val[:, None]
        val_data.append(ecg)
    val_data = np.array(val_data)

    test_data = []
    for id in test_ids:
        ecg = np.load(path + id + '.npy', allow_pickle=True).item()
        ecg = ecg['waveforms']['ecg_median']
        ecg = ecg[:,150:-50]
        ecg = signal.resample(ecg, 200, axis=1)
        max_val = np.max(np.abs(ecg), axis=1)
        if np.sum(max_val) > 0:
            ecg = ecg / max_val[:, None]
        test_data.append(ecg)
    test_data = np.array(test_data)

    return train_data, train_outcomes, val_data, val_outcomes, test_data, test_outcomes, train_ids, val_ids, test_ids


path = 'ecgs/'
selected_outcome = 'acs' # 'omi'

X_train, y_train, X_val, y_val, X_test, y_test, ID_train, ID_val, ID_test = get_data(path, selected_outcome)


#ECG Features
full_df = pd.read_csv('features.csv')
ID = full_df.iloc[:, 0]
X = full_df.set_index(ID)
X['sex'].replace(['Male', 'Female'],[1, 0], inplace=True)
X.drop(X.columns[0], axis = 1, inplace = True)
X.fillna(0, inplace=True)
cols = X.columns

X_features_train = X.loc[ID_train].values
X_features_val = X.loc[ID_val].values
X_features_test = X.loc[ID_test].values

X_features_train = pd.DataFrame(data=X_features_train, columns = cols, index=ID_train)
X_features_val = pd.DataFrame(data=X_features_val, columns = cols, index=ID_val)
X_features_test = pd.DataFrame(data=X_features_test, columns = cols, index=ID_test)

#ECGSMARTNET
class ResidualBlock2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel=3):
        super(ResidualBlock2D,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(1,kernel), stride=(1,stride), padding=(0, kernel//2), bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(1,kernel), stride=(1,1), padding=(0, kernel//2), bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(1,1), stride=(1,stride), bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out1 = self.conv1(x)
        out1 = self.bn1(out1)
        out1 = self.relu(out1)

        out1 = self.conv2(out1)
        out1 = self.bn2(out1)
        out1 = out1 + self.shortcut(x)
        out1 = self.relu(out1)

        return out1

class ECGSMARTNET(nn.Module):
    def __init__(self, num_classes=2, kernel=7, kernel1=3, num_leads=12, dropout=False):
        super(ECGSMARTNET, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=(1,kernel), stride=(1,2), padding=(0,kernel//2), bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,2), padding=(0,1))

        self.layer1 = self.make_layer(64, 2, stride=1, kernel=kernel1)
        self.layer2 = self.make_layer(128, 2, stride=2, kernel=kernel1)
        self.layer3 = self.make_layer(256, 2, stride=2, kernel=kernel1)
        self.layer4 = self.make_layer(512, 2, stride=2, kernel=kernel1)

        self.conv2 = nn.Conv2d(512, 512, kernel_size=(num_leads,1), stride=(1,1), padding=(0,0), bias=False)
        self.bn2 = nn.BatchNorm2d(512)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, num_classes)
        self.dropout = dropout
        self.do = nn.Dropout(p=0.2)

    def make_layer(self, out_channels, num_blocks, stride, kernel):
        layers = []

        layers.append(ResidualBlock2D(self.in_channels, out_channels, stride, kernel))
        self.in_channels = out_channels

        for _ in range(1, num_blocks):
            layers.append(ResidualBlock2D(self.in_channels, out_channels, 1, kernel))

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.maxpool(out)

        out = self.layer1(out)

        out = self.layer2(out)

        out = self.layer3(out)

        out = self.layer4(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        if self.dropout:
            out = self.do(out)

        return out

#MLP Model
class MLPModel(torch.nn.Module):
  def __init__(self, input_size, num_classes):
        super(MLPModel, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

  def forward(self, x):
        x = self.fc(x)
        return x


#Models
rf_estimators = 50
ecgsmartnet = ECGSMARTNET()
mlp = MLPModel(input_size=2*2, num_classes=2)
rf = RandomForestClassifier(class_weight='balanced_subsample', criterion='gini', n_jobs=-1, random_state=42,
                                                            max_features='sqrt',
                                                            n_estimators=50,
                                                            min_samples_split=0.001,
                                                            min_samples_leaf=0.001,
                                                            min_impurity_decrease=0,
                                                            bootstrap=True,
                                                            ccp_alpha=0.001,
                                                            max_samples=0.75,
                                                            oob_score=True)


#Training
num_epochs = 200
batch_size = 32
update_rf_every = 5
learning_rate1 = 1e-3
learning_rate2 = 1e-4
wd = 1e-2
patience = 10
best_val_metric = 0
patience_counter = 0
alpha = 2/3

ecgsmartnet = ecgsmartnet.to(device)
mlp = mlp.to(device)
params = list(ecgsmartnet.parameters()) + list(mlp.parameters()) #combined params

optimizer_ecg1 = torch.optim.Adam(ecgsmartnet.parameters(), lr=learning_rate1, weight_decay=wd)
optimizer_ecg2 = torch.optim.Adam(ecgsmartnet.parameters(), lr=learning_rate2, weight_decay=wd)
optimizer_mlp = torch.optim.Adam(mlp.parameters(), lr=6e-4, weight_decay=1e-3)
criterion = torch.nn.CrossEntropyLoss()

#Fit RF
rf.fit(X_features_train, y_train)

def create_balanced_batch(X, y, batch_size):
    class_0_indices = torch.where(y == 0)[0]
    class_1_indices = torch.where(y == 1)[0]

    samples_per_class = batch_size // 2
    samples_per_class = min(samples_per_class, len(class_0_indices), len(class_1_indices))

    class_0_sample = class_0_indices[torch.randint(len(class_0_indices), (samples_per_class,))]
    class_1_sample = class_1_indices[torch.randint(len(class_1_indices), (samples_per_class,))]

    balanced_indices = torch.cat([class_0_sample, class_1_sample])

    return balanced_indices[:batch_size]


for epoch in range(num_epochs):
    ecgsmartnet.train()
    mlp.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        indices = create_balanced_batch(X_train, y_train, batch_size)
        IDs = ID_train[indices]
        batch_X, batch_y = X_train[indices], y_train[indices]
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        X_features_batch = X_features_train.loc[IDs] #create batches for features

        outputs_ecg = ecgsmartnet(batch_X) #predictions from ECG-SMART-NET
        outputs_rf = rf.predict_proba(X_features_batch) #predictions from RF
        outputs_rf = torch.tensor(outputs_rf, dtype=torch.float32).to(device)
        inputs_mlp = torch.cat((outputs_ecg, outputs_rf), dim=1)
        outputs = mlp(inputs_mlp) #concatenated predictions into MLP


        batch_y = batch_y.squeeze()
        loss = criterion(outputs, batch_y)

        if epoch == 0:
          optimizer_ecg1.zero_grad()
          optimizer_mlp.zero_grad()

          loss.backward()

          optimizer_ecg1.step()
          optimizer_mlp.step()


        else:
          optimizer_ecg2.zero_grad()
          optimizer_mlp.zero_grad()

          loss.backward()

          optimizer_ecg2.step()
          optimizer_mlp.step()



        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")


    
    ecgsmartnet.eval()
    mlp.eval()
    val_loss = 0
    with torch.no_grad():
        #Validation
        X_val, y_val = X_val.to(device), y_val.to(device)
        val_outputs_ecg = ecgsmartnet(X_val)
        val_outputs_rf = rf.predict_proba(X_features_val)
        val_outputs_rf = torch.tensor(val_outputs_rf, dtype=torch.float32).to(device)
        inputs_mlp = torch.cat((val_outputs_ecg, val_outputs_rf), dim=1)
        val_outputs = mlp(inputs_mlp)
        y_val_squeezed = y_val.squeeze()
        loss = criterion(val_outputs, y_val_squeezed)
        val_loss += loss.item()

        val_outputs_ecg = sigmoid(val_outputs_ecg)
        val_outputs = sigmoid(val_outputs)
        y_val_np = y_val.cpu().numpy()
        val_outputs = val_outputs.cpu().numpy()
        val_outputs_ecg = val_outputs_ecg.cpu().numpy()
        auc_val_ecg = roc_auc_score(y_val_np, val_outputs_ecg[:,1])
        ap_val_ecg = average_precision_score(y_val_np, val_outputs_ecg[:,1])
        auc_val = roc_auc_score(y_val_np, val_outputs[:,1])
        ap_val = average_precision_score(y_val_np, val_outputs[:,1])

        val_metric = alpha*auc_val + (1 - alpha) * ap_val #compute validation metric

        #Test
        X_test, y_test = X_test.to(device), y_test.to(device)
        test_outputs_ecg = ecgsmartnet(X_test)
        test_outputs_rf = rf.predict_proba(X_features_test)
        test_outputs_rf = torch.tensor(test_outputs_rf, dtype=torch.float32).to(device)

        inputs_mlp =  torch.cat((test_outputs_ecg, test_outputs_rf), dim=1)
        test_outputs = mlp(inputs_mlp)

        test_outputs_ecg = sigmoid(test_outputs_ecg)
        test_outputs = sigmoid(test_outputs)
        y_test_np = y_test.cpu().numpy()
        test_outputs = test_outputs.cpu().numpy()
        test_outputs_ecg = test_outputs_ecg.cpu().numpy()
        auc_ecg = roc_auc_score(y_test_np, test_outputs_ecg[:,1])
        ap_ecg = average_precision_score(y_test_np, test_outputs_ecg[:,1])
        auc_test = roc_auc_score(y_test_np, test_outputs[:,1])
        ap_test = average_precision_score(y_test_np, test_outputs[:,1])


    if val_metric > best_val_metric:
        best_val_metric = val_metric
        patience_counter = 0
        #save model
        torch.save(ecgsmartnet.state_dict(), "best_ecgsmartnet_model_alternate.pth")
        torch.save(mlp.state_dict(), "best_mlp_model_alternate.pth")

    else:
        patience_counter += 1
        print(f"No drop in val loss for {patience_counter} epochs.")

    if patience_counter >= patience:
        print("EarlyStopping triggered.")
        break

    #log progress
    print(f"ECGSMARTNET Val AUC: {auc_val_ecg:.4f}, AP: {ap_val_ecg:.4f}")
    print(f"Validation AUC: {auc_val:.4f}, AP: {ap_val:.4f}, Metric: {val_metric:.4f}, Loss: {val_loss:.4f}")
    print(f"ECGSMARTNET Test AUC: {auc_ecg:.4f}, AP: {ap_ecg:.4f}")
    print(f"Test AUC: {auc_test:.4f}, AP: {ap_test:.4f}")

