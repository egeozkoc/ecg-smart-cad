import torch
import numpy as np
from captum.attr import Saliency, GuidedGradCam
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.ticker import MultipleLocator
import pandas as pd
from matplotlib.cm import ScalarMappable

def get_density_plot(y_pred, y):
    # plot a bar graph of the predictions for the two classes

    y_pred_0 = y_pred[y == 0]
    y_pred_1 = y_pred[y == 1]

    plt.hist([y_pred_0, y_pred_1], bins=50, label=['No OMI', 'OMI'], density=True)
    plt.legend()
    plt.show()

def get_prediction_probs(model, test_ids, test_omi):
    probs = []
    for id in test_ids:
        id = id.replace('median_beats', 'median_beats_nolinenoise')
        ecg = np.load(id, allow_pickle=True).item()['ecg']
        ecg = ecg[:, 150:-50] / 1000

        mean_ecg = np.mean(ecg, axis=-1)
        std_ecg = np.std(ecg, axis=-1)
        std_ecg[std_ecg == 0] = 1
        ecg = (ecg - mean_ecg[:, np.newaxis]) / std_ecg[:, np.newaxis]

        ecg = torch.Tensor(ecg)
        ecg = ecg.unsqueeze(0)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ecg = ecg.to(device)

        predicted = model(ecg)
        predicted = torch.softmax(predicted, dim=-1).squeeze().cpu().detach().numpy()
        probs.append(predicted[1])
    probs = np.array(probs)

    get_density_plot(probs, test_omi)

    idx = np.argsort(probs)
    probs = probs[idx]
    test_ids = test_ids[idx]
    test_omi = test_omi[idx]

    test_ids = test_ids[test_omi == 0]

    #only keep test_ids that end in '-1.npy'
    test_ids = [id for id in test_ids if id[-6:] == '-1.npy']

    # test_ids = test_ids[(test_omi == 1) & (probs > 0.57)] 
    
    # test_ids = test_ids[::-1]
    #randomly shuffle the test ids
    # np.random.shuffle(test_ids)

    test_ids = ['../../../../ECG Datasets/Pitt Data/median_beats\\sp2026-3.npy']

    return test_ids

resolution = 1

def on_resize(event):
    fig = event.canvas.figure
    fig.set_size_inches(8* resolution, 9 * resolution)  # Set the desired fixed size

num_leads = 12
target_class = 1

if num_leads == 12:
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
elif num_leads == 8:
    leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

model = torch.load('final models/model10_ecgsmartnet_nolinenoise.pt')
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)
model.eval()
saliency = Saliency(model)

folds = np.load('omi_folds_fixed.npy', allow_pickle=True).item()

val_ids = folds['fold_ids'][9]
val_omi = folds['fold_omi'][9]
test_ids = folds['fold_ids'][0]
test_omi = folds['fold_omi'][0]

# val_ids = val_ids[val_omi == target_class]
# test_ids = test_ids[test_omi == target_class]

# test_ids = test_ids[test_omi == target_class]
# test_omi = test_omi[test_omi == target_class]

fs = 500

chosen_ids = get_prediction_probs(model, test_ids, test_omi)



for id in chosen_ids:
    print(id)
    id = id.replace('median_beats', 'median_beats_nolinenoise')
    ecg = np.load(id, allow_pickle=True).item()['ecg']
    ecg = ecg[:, 150:-50] / 1000

    ecg1 = ecg.copy()

    mean_ecg = np.mean(ecg, axis=-1)
    std_ecg = np.std(ecg, axis=-1)
    std_ecg[std_ecg == 0] = 1
    ecg = (ecg - mean_ecg[:, np.newaxis]) / std_ecg[:, np.newaxis]


    if num_leads == 8:
        ecg = ecg[[0,1,6,7,8,9,10,11], :]
        ecg1 = ecg1[[0,1,6,7,8,9,10,11], :]

    fiducial_id = id.replace('median_beats', 'fiducials')
    fiducials = np.load(fiducial_id)
    fiducials -= 150
    fiducials[fiducials < 0] = 0
    ecg_rms = np.sqrt(np.mean(ecg**2, axis=0))

    ecg = torch.Tensor(ecg)
    ecg = ecg.unsqueeze(0)
    ecg = ecg.to(device)
    ecg.requires_grad_()

    predicted = model(ecg)
    predicted = torch.softmax(predicted, dim=-1).squeeze().cpu().detach().numpy()
    predicted = predicted[1]
    print(predicted)

    saliency_map = saliency.attribute(ecg, target=target_class, abs = True)
    saliency_map = saliency_map.squeeze().cpu().detach().numpy()
    # saliency_map[saliency_map < 0] = 0

    contributions = np.sum(saliency_map, axis=1) / np.sum(saliency_map) * 100
    print(contributions)
    
    for i in range(0,400,20):
        for j in range(num_leads):
            saliency_map[j, i:i+20] = np.mean(saliency_map[j, i:i+20], axis=-1)

    saliency_min = np.min(saliency_map)
    saliency_max = np.max(saliency_map)
    saliency_map = (saliency_map - saliency_min) / (saliency_max - saliency_min)

    color_points = [
        (0.0, (1,0.9,0)),
        (0.275, (1,.45,0)),
        (0.55, (1,0,0)),
        (1.0, (0.25,0,0))
    ]
    cmap = LinearSegmentedColormap.from_list('custom_cmap', color_points)
    color0 = (1,0.9,0)

    start = fiducials[0]
    end = fiducials[5]
    
    saliency_colors = cmap(saliency_map)

    ecg = ecg.squeeze().cpu().detach().numpy()


    fig, ax = plt.subplots(figsize=(8*resolution, 9*resolution))
    fig.canvas.mpl_connect('resize_event', on_resize)
    offset = 0
    x = np.arange(0,400,1)
    for i in range(0,12,3):
        ax.plot(x + offset, ecg1[i], color='k', linewidth=2*resolution)
        ax.plot(x + offset, ecg1[i+1] - 3, color='k', linewidth=2*resolution)
        ax.plot(x + offset, ecg1[i+2] - 6, color='k', linewidth=2*resolution)
        ax.text(x=10 + offset, y=1, s=leads[i], fontsize=20*resolution, weight='bold')
        ax.text(x=10 + offset, y=1-3, s=leads[i+1], fontsize=20*resolution, weight='bold')
        ax.text(x=10 + offset, y=1-6, s=leads[i+2], fontsize=20*resolution, weight='bold')

        offset += 400

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
    # plt.title(id.split('\\')[-1][:-4] + " " + str(predicted), fontsize=20)
    fig.tight_layout()
    # add colorbar on x axis
    # cmappable = ScalarMappable(norm=Normalize(0,1), cmap = cmap)
    # cbar = plt.colorbar(cmappable, ax=ax, orientation='horizontal')
    plt.show()
    # if predicted > 0.57:
    #     plt.savefig('C:/Users/nater/OneDrive/Desktop/new_saliency_maps/false_positive/{}_gray.png'.format(id.split('\\')[-1][:-4]), dpi=300)
    # else:
    #     plt.savefig('C:/Users/nater/OneDrive/Desktop/new_saliency_maps/true_negative/{}_gray.png'.format(id.split('\\')[-1][:-4]), dpi=300)
    # plt.close()


    fig, ax = plt.subplots(figsize=(8*resolution, 9*resolution))
    fig.canvas.mpl_connect('resize_event', on_resize)
    offset = 0
    x = np.arange(0,400,1)
    for i in range(0,12,3):
        if i < num_leads:
            colors_lead = saliency_colors[i, :, :3]  # Extract the RGB colors for the lead
            for j in range(400 - 1):
                if j <= start or j >= end:
                    ax.plot(x[j:j+2] + offset, ecg1[i, j:j+2], color=color0, linewidth=2*resolution)
                else:
                    ax.plot(x[j:j+2] + offset, ecg1[i, j:j+2], color=colors_lead[j:j+1], linewidth=2*resolution)
            ax.text(x=10 + offset, y=1, s=leads[i], fontsize=20*resolution, weight='bold')
            ax.text(x=10 + offset, y=1-.3, s='{0:.0f}%'.format(contributions[i]), fontsize=16*resolution, weight='bold')

        if i+1 < num_leads:
            colors_lead = saliency_colors[i+1, :, :3]
            for j in range(400 - 1):
                if j <= start or j >= end:
                    ax.plot(x[j:j+2] + offset, ecg1[i+1, j:j+2] - 3, color=color0, linewidth=2*resolution)
                else:
                    ax.plot(x[j:j+2] + offset, ecg1[i+1, j:j+2] - 3, color=colors_lead[j:j+1], linewidth=2*resolution)
            ax.text(x=10 + offset, y=1-3, s=leads[i+1], fontsize=20*resolution, weight='bold')
            ax.text(x=10 + offset, y=1-3.3, s='{0:.0f}%'.format(contributions[i+1]), fontsize=16*resolution, weight='bold')

        if i+2 < num_leads:
            colors_lead = saliency_colors[i+2, :, :3]
            for j in range(400 - 1):
                if j <= start or j >= end:
                    ax.plot(x[j:j+2] + offset, ecg1[i+2, j:j+2] - 6, color=color0, linewidth=2*resolution)
                else: 
                    ax.plot(x[j:j+2] + offset, ecg1[i+2, j:j+2] - 6, color=colors_lead[j:j+1], linewidth=2*resolution)
            ax.text(x=10 + offset, y=1-6, s=leads[i+2], fontsize=20*resolution, weight='bold')
            ax.text(x=10 + offset, y=1-6.3, s='{0:.0f}%'.format(contributions[i+2]), fontsize=16*resolution, weight='bold')

        offset += 400

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
    # plt.title(id.split('\\')[-1][:-4] + " " + str(predicted), fontsize=20)
    # plt.title(id.split('\\')[-1][:-4], fontsize=20)
    fig.tight_layout()
    # add colorbar on x axis
    # cmappable = ScalarMappable(norm=Normalize(0,1), cmap = cmap)
    # cbar = plt.colorbar(cmappable, ax=ax, orientation='horizontal')
    plt.show()
    # if predicted > 0.57:
    #     plt.savefig('C:/Users/nater/OneDrive/Desktop/new_saliency_maps/false_positive/{}.png'.format(id.split('\\')[-1][:-4]), dpi=300)
    # else:
    #     plt.savefig('C:/Users/nater/OneDrive/Desktop/new_saliency_maps/true_negative/{}.png'.format(id.split('\\')[-1][:-4]), dpi=300)
    # plt.close()




print('done')