import torch
import numpy as np
from captum.attr import Saliency
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MultipleLocator
import pandas as pd

# train the model 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import roc_auc_score, auc, accuracy_score, recall_score, precision_score, f1_score, average_precision_score
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
from scipy import signal
from ecg_models import *
from torch.utils.tensorboard import SummaryWriter
from ecg import ECG

resolution = 1
# def on_resize(event):
#     fig = event.canvas.figure
#     fig.set_size_inches(9* resolution, 8 * resolution)  # Set the desired fixed size

target_class = 1
leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
fs = 500

# model = torch.load('models/temporal2d_omi.pt')
model = torch.load('models/ecgsmartnet500_omi.pt', weights_only=False)
model.cpu()
model.eval()
saliency = Saliency(model)

test_data = pd.read_csv('test_data_omi.csv')
test_ids = test_data['id'][test_data['outcome'] == target_class].to_list()

average_ecg = np.zeros((12,int(fs*0.8)))
average_saliency_map = np.zeros((12,int(fs*0.8)))
average_saliency_iqrs = np.zeros(12)
average_saliency_tqrs = np.zeros(12)
average_saliency_it = np.zeros(12)
average_saliency_tt = np.zeros(12)
average_saliency_pr = np.zeros(12)


start = 0
end = 0
path = '../ecgs100/'

for id in test_ids:
    ecg = np.load(path + id + '.npy', allow_pickle=True).item()
    fiducials = ecg.fiducials['global']
    fiducials -= 150
    fiducials[fiducials < 0] = 0
    ecg = ecg.waveforms['ecg_median']
    ecg = ecg[:,150:-50]
    if fs != 500:
        ecg = signal.resample(ecg, int(fs*0.8), axis=1)
    average_ecg += (ecg / 1000)
    max_val = np.max(np.abs(ecg), axis=1)
    ecg = ecg / max_val[:, None]

    start += fiducials[0]
    end += fiducials[5]

    ecg = torch.Tensor(ecg)
    ecg = ecg.unsqueeze(0)
    ecg = ecg.unsqueeze(0)
    ecg.cpu()
    ecg.requires_grad_()

    predicted = model(ecg)
    predicted = torch.softmax(predicted, dim=-1).squeeze().cpu().detach().numpy()
    predicted = predicted[target_class]

    saliency_map = saliency.attribute(ecg, target=target_class, abs = True)
    saliency_map = saliency_map.squeeze().cpu().detach().numpy()
    average_saliency_map += saliency_map

start /= len(test_ids)
end /= len(test_ids)
start /= 2
end /= 2

start = int(start)
end = int(end)

average_ecg /= len(test_ids)
average_saliency_map /= len(test_ids)

contributions = np.sum(average_saliency_map, axis=1) / np.sum(average_saliency_map) * 100
print(contributions)

for i in range(0,int(fs*0.8),int(fs*0.04)):
    for j in range(12):
        average_saliency_map[j, i:i+int(fs*0.04)] = np.mean(average_saliency_map[j, i:i+int(fs*0.04)])

average_saliency_map = (average_saliency_map - np.min(average_saliency_map[:,start:end])) / (np.max(average_saliency_map[:,start:end]) - np.min(average_saliency_map[:,start:end]))

# for i in range(12):
#     average_saliency_map[i,start:end] = (average_saliency_map[i,start:end] - np.min(average_saliency_map[i,start:end])) / (np.max(average_saliency_map[i,start:end]) - np.min(average_saliency_map[i,start:end]))

color_points = [
        (0.0, (1,1,0)),
        (2/7, (1,.5,0)),
        (4/7, (1,0,0)),
        (6/7, (.5,0,0)),
        (1.0, (.25,0,0))
    ]

cmap = LinearSegmentedColormap.from_list('custom_cmap', color_points)
color0 = (1,1,0)

saliency_colors = cmap(average_saliency_map)

start = 0
end = int(fs*0.8)


fig, ax = plt.subplots(figsize=(9*resolution, 8*resolution))
# fig.canvas.mpl_connect('resize_event', on_resize)
offset = 0
x = np.arange(0,int(fs*0.8),1)
for i in range(0,12,3):
    colors_lead = saliency_colors[i, :, :3]  # Extract the RGB colors for the lead
    for j in range(int(fs*0.8) - 1):
        if j <= start or j >= end:
            ax.plot(x[j:j+2] + offset, average_ecg[i, j:j+2], color=color0, linewidth=2*resolution)
        else:    
            ax.plot(x[j:j+2] + offset, average_ecg[i, j:j+2], color=colors_lead[j:j+1], linewidth=2*resolution)
    ax.text(x=10 + offset, y=1, s=leads[i], fontsize=20*resolution, weight='bold')
    ax.text(x=10 + offset, y=1-.3, s='{0:.0f}%'.format(contributions[i]), fontsize=16*resolution, weight='bold')

    colors_lead = saliency_colors[i+1, :, :3]
    for j in range(int(fs*0.8) - 1):
        if j <= start or j >= end:
            ax.plot(x[j:j+2] + offset, average_ecg[i+1, j:j+2] - 3, color=color0, linewidth=2*resolution)
        else:
            ax.plot(x[j:j+2] + offset, average_ecg[i+1, j:j+2] - 3, color=colors_lead[j:j+1], linewidth=2*resolution)
    ax.text(x=10 + offset, y=1-3, s=leads[i+1], fontsize=20*resolution, weight='bold')
    ax.text(x=10 + offset, y=1-3.3, s='{0:.0f}%'.format(contributions[i+1]), fontsize=16*resolution, weight='bold')

    colors_lead = saliency_colors[i+2, :, :3]
    for j in range(int(fs*0.8) - 1):
        if j <= start or j >= end:
            ax.plot(x[j:j+2] + offset, average_ecg[i+2, j:j+2] - 6, color=color0, linewidth=2*resolution)
        else:
            ax.plot(x[j:j+2] + offset, average_ecg[i+2, j:j+2] - 6, color=colors_lead[j:j+1], linewidth=2*resolution)
    ax.text(x=10 + offset, y=1-6, s=leads[i+2], fontsize=20*resolution, weight='bold')
    ax.text(x=10 + offset, y=1-6.3, s='{0:.0f}%'.format(contributions[i+2]), fontsize=16*resolution, weight='bold')

    offset += int(fs*0.8)

ax.grid(visible=True, which='major', linestyle='-', linewidth=0.5*resolution, color=(0.1,0.1,0.1), alpha=1)
ax.grid(visible=True, which='minor', linestyle='-', linewidth=0.5*resolution, color=(0.1,0.1,0.1), alpha=0.2)

minor_locator = MultipleLocator(0.04*fs)
major_locator = MultipleLocator(0.2*fs)
ax.xaxis.set_minor_locator(minor_locator)
ax.xaxis.set_major_locator(major_locator)

minor_locator = MultipleLocator(0.1)
major_locator = MultipleLocator(0.5)
ax.yaxis.set_minor_locator(minor_locator)
ax.yaxis.set_major_locator(major_locator)
plt.xticks(visible=False)
plt.yticks(visible=False)
ax.set_facecolor((0.85,0.85,0.85))

plt.axvline(x=0.8*fs, color='k', linestyle='-', linewidth=2*resolution)
plt.axvline(x=0.8*fs*2, color='k', linestyle='-', linewidth=2*resolution)
plt.axvline(x=0.8*fs*3, color='k', linestyle='-', linewidth=2*resolution)
plt.xlim((0, 0.8*fs*4))
plt.ylim((-7.5, 1.5))
# plt.title('Average OMI', fontsize=20)

# cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, orientation='horizontal')
# cbar.ax.set_xticklabels(['Least Important', ' ', ' ', ' ', ' ', 'Most Important'], fontsize=20*resolution, weight='bold')

fig.tight_layout()
plt.show()
# plt.savefig('final models/saliency_12lead_2d.png', dpi=300)


############################################################################################
average_saliency_map = np.mean(average_saliency_map, axis=0)
average_saliency_map = (average_saliency_map - np.min(average_saliency_map[start:end])) / (np.max(average_saliency_map[start:end]) - np.min(average_saliency_map[start:end]))
saliency_colors = cmap(average_saliency_map)

average_rms = np.sqrt(np.mean(average_ecg**2, axis=0))


fig, ax = plt.subplots(figsize=(8*resolution, 4*resolution))
# fig.canvas.mpl_connect('resize_event', on_resize)
offset = 0
x = np.arange(0,int(fs*0.8),1)
colors_lead = saliency_colors[:, :3]  # Extract the RGB colors for the lead
for j in range(int(fs*0.8) - 1):
    if j <= start or j >= end:
        ax.plot(x[j:j+2] + offset, average_rms[j:j+2], color=color0, linewidth=2*resolution)
    else:    
        ax.plot(x[j:j+2] + offset, average_rms[j:j+2], color=colors_lead[j:j+1], linewidth=2*resolution)
ax.text(x=10 + offset, y=.75, s='RMS', fontsize=20*resolution, weight='bold')

ax.grid(visible=True, which='major', linestyle='-', linewidth=0.5*resolution, color=(0.1,0.1,0.1), alpha=1)
ax.grid(visible=True, which='minor', linestyle='-', linewidth=0.5*resolution, color=(0.1,0.1,0.1), alpha=0.2)

minor_locator = MultipleLocator(0.04*fs)
major_locator = MultipleLocator(0.2*fs)
ax.xaxis.set_minor_locator(minor_locator)
ax.xaxis.set_major_locator(major_locator)

minor_locator = MultipleLocator(0.1)
major_locator = MultipleLocator(0.5)
ax.yaxis.set_minor_locator(minor_locator)
ax.yaxis.set_major_locator(major_locator)
plt.xticks(visible=False)
plt.yticks(visible=False)
ax.set_facecolor((0.85,0.85,0.85))

plt.xlim((0, 0.8*fs))
plt.ylim((0, 1))
# plt.title('Average OMI', fontsize=20)

# cbar = fig.colorbar(plt.cm.ScalarMappable(cmap=cmap), ax=ax, orientation='horizontal')
# cbar.ax.set_xticklabels(['Least Important', ' ', ' ', ' ', ' ', 'Most Important'], fontsize=20*resolution, weight='bold')

fig.tight_layout()
plt.show()


print('done')