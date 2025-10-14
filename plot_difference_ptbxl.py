import torch
import numpy as np
from captum.attr import Saliency
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
        ecg = np.load(id, allow_pickle=True).item()['ecg']
        ecg = ecg[:, 150:-50] / 1000

        max_val = np.max(np.abs(ecg), axis=-1)
        ecg = ecg.T
        ecg /= max_val
        ecg = ecg.T

        ecg[3, :] *= -1

        if num_leads == 8:
            ecg = ecg[[0, 1, 6, 7, 8, 9, 10, 11], :]

        ecg = torch.Tensor(ecg)
        ecg = ecg.unsqueeze(0)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        ecg = ecg.to(device)
        # ecg.requires_grad_()

        predicted = model(ecg)
        predicted = torch.softmax(predicted, dim=-1).squeeze().cpu().detach().numpy()
        probs.append(predicted[1])
    probs = np.array(probs)

    get_density_plot(probs, test_omi)

    idx = np.argsort(probs)
    probs = probs[idx]
    test_ids = test_ids[idx]
    test_omi = test_omi[idx]

    # get top 3 and bottom 3 predictions
    # top_idx = np.where(test_omi == 1)[0][0:20]
    bottom_idx = np.where(test_omi == 0)[0][0:5]

    top_idx = np.where((test_omi == 1) & (probs < 0.06))[0]

    # idx = np.argsort(test_ids[top_idx])
    # probs = probs[top_idx][idx]
    # top_ids = test_ids[top_idx][idx]

    # print(top_ids)
    # print(probs)

    top_ids = test_ids[top_idx]
    bottom_ids = test_ids[bottom_idx]

    # print(top_ids)
    # print(probs[top_idx])

    return top_ids, bottom_ids

resolution = 1

# def on_resize(event):
#     fig = event.canvas.figure
#     fig.set_size_inches(9* resolution, 8 * resolution)  # Set the desired fixed size

num_leads = 12
target_class = 1

if num_leads == 12:
    leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
elif num_leads == 8:
    leads = ['I', 'II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# model = torch.load('final analysis/final models/model_12lead_2d.pt')
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# model.to(device)
# model.eval()
# saliency = Saliency(model)

# fold = np.load('final analysis/omi_folds.npy', allow_pickle=True).item()

# test_ids = fold['test_ids']
# test_omi = fold['test_omi']

fs = 500

# top_ids, bottom_ids = get_prediction_probs(model, test_ids, test_omi)

# if target_class == 1:
#     chosen_ids = top_ids
# else:
#     chosen_ids = bottom_ids

prefix_pitt = 'C:/Users/nater/OneDrive/Documents/College/Research/ECG/ECG Datasets/Pitt Data/median_beats_nolinenoise/'
prefix_medic = 'C:/Users/nater/OneDrive/Documents/College/Research/ECG/ECG Datasets/MEDIC Data/median_beats_nolinenoise/'

# False Negatives
# chosen_ids_pitt = ['sp2026-1.npy', 'sp1048-1.npy', 'sp1048-2.npy', 'sp1048-3.npy', 'sp1911-1.npy', 'sp2026-2.npy', 'sp2026-3.npy', 'sp3307-2.npy']
# chosen_ids_pitt = ['sp1898-1.npy']
# chosen_ids_medic = ['me1443-1.npy', 'me2214-1.npy', 'me2214-2.npy', 'me2214-3.npy', 'me3720-1.npy', 'me3720-2.npy', 'me4495-1.npy', 'me4495-2.npy']

# # True Poitives
chosen_ids_pitt = []
chosen_ids_medic = ['me1500-1.npy', 'me1367-1.npy', 'me1367-2.npy', 'me1500-2.npy', 'me1500-3.npy', 'me1623-1.npy', 'me1789-1.npy', 'me1789-2.npy', 'me1940-1.npy']

chosen_ids_pitt = [prefix_pitt + id for id in chosen_ids_pitt]
chosen_ids_medic = [prefix_medic + id for id in chosen_ids_medic]
chosen_ids = chosen_ids_pitt + chosen_ids_medic


ecg_ref = np.load('ecg_avg_norm.npy', allow_pickle=True).item()['mean']
ecg_ref = ecg_ref[:, 150:-50] / 1000
ecg_ref_max = np.load('ecg_avg_norm.npy', allow_pickle=True).item()['max']
ecg_ref_max = ecg_ref_max[:, 150:-50] / 1000
ecg_ref_min = np.load('ecg_avg_norm.npy', allow_pickle=True).item()['min']
ecg_ref_min = ecg_ref_min[:, 150:-50] / 1000
ecg_ref_pr = np.load('ecg_avg_norm.npy', allow_pickle=True).item()['pr']
ecg_ref_qrs = np.load('ecg_avg_norm.npy', allow_pickle=True).item()['qrs']
ecg_ref_stt = np.load('ecg_avg_norm.npy', allow_pickle=True).item()['stt']

for id in chosen_ids:
    print(id)
    ecg = np.load(id, allow_pickle=True).item()['ecg']
    ecg = ecg[:, 150:-50] / 1000

    fiducial_id = id.replace('median_beats', 'fiducials')
    fiducials = np.load(fiducial_id)
    fiducials -= 150
    fiducials[fiducials < 0] = 0

    color_points = [
        (0.0, (1,1,0)),
        (2/7, (1,.5,0)),
        (4/7, (1,0,0)),
        (6/7, (.5,0,0)),
        (1.0, (.25,0,0))
    ]

    cmap = LinearSegmentedColormap.from_list('custom_cmap', color_points)
    # color0 = (1,0.9,0)
    color0 = (0.5,0.5,0.5)

    start = fiducials[0] - 20
    if start < 0:
        start = 0
    end = fiducials[5] + 20
    if end > 400:
        end = 400


    pr = np.abs(ecg[:,fiducials[0]:fiducials[2]] - ecg_ref[:, fiducials[0]:fiducials[2]])
    qrs = np.abs(ecg[:,fiducials[2]:fiducials[3]] - ecg_ref[:, fiducials[2]:fiducials[3]])
    st = np.abs(ecg[:,fiducials[3]:fiducials[5]] - ecg_ref[:, fiducials[3]:fiducials[5]])

    pr = (pr.T / ecg_ref_pr).T
    qrs = (qrs.T / ecg_ref_qrs).T
    st = (st.T / ecg_ref_stt).T

    pr /= np.max(pr)
    qrs /= np.max(qrs)
    st /= np.max(st)

    pr *= 0.5
    st *= 1.5

    saliency_map = np.zeros((12, 400))
    saliency_map[:, fiducials[0]:fiducials[2]] = pr
    saliency_map[:, fiducials[2]:fiducials[3]] = qrs
    saliency_map[:, fiducials[3]:fiducials[5]] = st

    contributions = np.empty(12)
    for i in range(12):
        bad = np.where((ecg[i,fiducials[0]:fiducials[5]] > ecg_ref_max[i,fiducials[0]:fiducials[5]]) | (ecg[i,fiducials[0]:fiducials[5]] < ecg_ref_min[i,fiducials[0]:fiducials[5]]))[0]

        bad_pr = np.where((ecg[i,fiducials[0]:fiducials[2]] > ecg_ref_max[i,fiducials[0]:fiducials[2]]) | (ecg[i,fiducials[0]:fiducials[2]] < ecg_ref_min[i,fiducials[0]:fiducials[2]]))[0]
        bad_qrs = np.where((ecg[i,fiducials[2]:fiducials[3]] > ecg_ref_max[i,fiducials[2]:fiducials[3]]) | (ecg[i,fiducials[2]:fiducials[3]] < ecg_ref_min[i,fiducials[2]:fiducials[3]]))[0]
        bad_st = np.where((ecg[i,fiducials[3]:fiducials[5]] > ecg_ref_max[i,fiducials[3]:fiducials[5]]) | (ecg[i,fiducials[3]:fiducials[5]] < ecg_ref_min[i,fiducials[3]:fiducials[5]]))[0]

        contributions[i] = len(bad_pr) / len(ecg[i,fiducials[0]:fiducials[2]]) * 0.5 + len(bad_qrs) / len(ecg[i,fiducials[2]:fiducials[3]]) + len(bad_st) / len(ecg[i,fiducials[3]:fiducials[5]]) * 1.5
        contributions[i] *= 100/3

        #normalize within each lead separately
        saliency_map[i,:] = (saliency_map[i,:] - np.min(saliency_map[i,fiducials[0]:fiducials[5]])) / (np.max(saliency_map[i,fiducials[0]:fiducials[5]]) - np.min(saliency_map[i,fiducials[0]:fiducials[5]]))
        saliency_map[i,:] *= (contributions[i] / 100)
    
    saliency_map = (saliency_map - np.min(saliency_map[:,fiducials[0]:fiducials[5]])) / (np.max(saliency_map[:,fiducials[0]:fiducials[5]]) - np.min(saliency_map[:,fiducials[0]:fiducials[5]]))
    
    saliency_colors = cmap(saliency_map)

    fig, ax = plt.subplots(figsize=(9*resolution, 8*resolution))
    # fig.canvas.mpl_connect('resize_event', on_resize)
    offset = 0

    x = np.arange(0,400,1)
    for i in range(0,12,3):
        if i < num_leads:
            colors_lead = saliency_colors[i, :, :3]  # Extract the RGB colors for the lead
            red = colors_lead[:,0]
            red = np.sum(red[red < 0.75])

            for j in range(400 - 1):
                if j <= start or j >= end:
                    # ax.plot(x[j:j+2] + offset, ecg[i, j:j+2], color=color0, linewidth=2*resolution, alpha=0.8)
                    pass
                else:
                    ax.plot(x[j:j+2] + offset, ecg[i, j:j+2], color=colors_lead[j:j+1], linewidth=2*resolution)

            ax.plot(x[:start]+offset, np.repeat(ecg[i, start],start), linewidth=2*resolution, color=color0)
            ax.plot(x[end:]+offset, np.repeat(ecg[i, end],400-end), linewidth=2*resolution, color=color0)


            
            if contributions[i] >= 50:
                ax.axvspan(xmin = offset, xmax= offset+400, ymin= 2/3, ymax=1,  color='blue', alpha=0.1)
                ax.text(x=10 + offset, y=1, s=leads[i], fontsize=20*resolution, weight='bold')
                ax.text(x=10 + offset, y=1-.3, s='{0:.0f}%'.format(contributions[i]), fontsize=16*resolution, weight='bold')
            else:
                ax.text(x=10 + offset, y=1, s=leads[i], fontsize=16*resolution)
                ax.text(x=10 + offset, y=1-.3, s='{0:.0f}%'.format(contributions[i]), fontsize=12*resolution)

        if i+1 < num_leads:
            colors_lead = saliency_colors[i+1, :, :3]
            red = colors_lead[:,0]
            red = np.sum(red[red < 0.75])

            for j in range(400 - 1):
                if j <= start or j >= end:
                    # ax.plot(x[j:j+2] + offset, ecg[i+1, j:j+2] - 3, color=color0, linewidth=2*resolution, alpha=0.8)
                    pass
                else:
                    ax.plot(x[j:j+2] + offset, ecg[i+1, j:j+2] - 3, color=colors_lead[j:j+1], linewidth=2*resolution)

            ax.plot(x[:start]+offset, np.repeat(ecg[i+1, start] - 3,start), linewidth=2*resolution, color=color0)
            ax.plot(x[end:]+offset, np.repeat(ecg[i+1, end] - 3,400-end), linewidth=2*resolution, color=color0)

            if contributions[i+1] >= 50:
                ax.axvspan(xmin = offset, xmax= offset+400, ymin= 1/3, ymax=2/3,  color='blue', alpha=0.1)
                ax.text(x=10 + offset, y=1-3, s=leads[i+1], fontsize=20*resolution, weight='bold')
                ax.text(x=10 + offset, y=1-3-.3, s='{0:.0f}%'.format(contributions[i+1]), fontsize=16*resolution, weight='bold')
            else:
                ax.text(x=10 + offset, y=1-3, s=leads[i+1], fontsize=16*resolution)
                ax.text(x=10 + offset, y=1-3-.3, s='{0:.0f}%'.format(contributions[i+1]), fontsize=12*resolution)

        if i+2 < num_leads:
            colors_lead = saliency_colors[i+2, :, :3]
            red = colors_lead[:,0]
            red = np.sum(red[red < 0.75])

            for j in range(400 - 1):
                if j <= start or j >= end:
                    # ax.plot(x[j:j+2] + offset, ecg[i+2, j:j+2] - 6, color=color0, linewidth=2*resolution, alpha=0.8)
                    pass
                else: 
                    ax.plot(x[j:j+2] + offset, ecg[i+2, j:j+2] - 6, color=colors_lead[j:j+1], linewidth=2*resolution)

            ax.plot(x[:start]+offset, np.repeat(ecg[i+2, start] - 6,start), linewidth=2*resolution, color=color0)
            ax.plot(x[end:]+offset, np.repeat(ecg[i+2, end] - 6,400-end), linewidth=2*resolution, color=color0)

            if contributions[i+2] >= 50:
                ax.axvspan(xmin = offset, xmax= offset+400, ymin= 0, ymax=1/3,  color='blue', alpha=0.1)
                ax.text(x=10 + offset, y=1-6, s=leads[i+2], fontsize=20*resolution, weight='bold')
                ax.text(x=10 + offset, y=1-6-.3, s='{0:.0f}%'.format(contributions[i+2]), fontsize=16*resolution, weight='bold')
            else:
                ax.text(x=10 + offset, y=1-6, s=leads[i+2], fontsize=16*resolution)
                ax.text(x=10 + offset, y=1-6-.3, s='{0:.0f}%'.format(contributions[i+2]), fontsize=12*resolution)
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
    plt.title(id.split('/')[-1][:-4], fontsize=20)
    fig.tight_layout()
    # add colorbar on x axis
    # cmappable = ScalarMappable(norm=Normalize(0,1), cmap = cmap)
    # cbar = plt.colorbar(cmappable, ax=ax, orientation='horizontal')
    plt.show()
    # plt.savefig('C:/Users/nater/OneDrive/Desktop/true positives diff norm/{}.png'.format(id.split('/')[-1][:-4]), dpi=300)
    # plt.close()

    # now plot RMS signal
    ecg_ref_rms = np.load('ecg_avg_norm.npy', allow_pickle=True).item()['mean_rms']
    ecg_ref_rms = ecg_ref_rms[150:-50] / 1000
    ecg_ref_rms_max = np.load('ecg_avg_norm.npy', allow_pickle=True).item()['max_rms']
    ecg_ref_rms_max = ecg_ref_rms_max[150:-50] / 1000
    ecg_ref_rms_min = np.load('ecg_avg_norm.npy', allow_pickle=True).item()['min_rms']
    ecg_ref_rms_min = ecg_ref_rms_min[150:-50] / 1000

    ecg_rms = np.sqrt(np.sum(ecg**2, axis=0))/np.sqrt(12)

    pr = np.abs(ecg_rms[fiducials[0]:fiducials[2]] - ecg_ref_rms[fiducials[0]:fiducials[2]])
    qrs = np.abs(ecg_rms[fiducials[2]:fiducials[3]] - ecg_ref_rms[fiducials[2]:fiducials[3]])
    st = np.abs(ecg_rms[fiducials[3]:fiducials[5]] - ecg_ref_rms[fiducials[3]:fiducials[5]])

    pr /= np.max(pr)
    qrs /= np.max(qrs)
    st /= np.max(st)

    pr *= 0.5
    st *= 1.5

    saliency_map = np.zeros((400))
    saliency_map[fiducials[0]:fiducials[2]] = pr
    saliency_map[fiducials[2]:fiducials[3]] = qrs
    saliency_map[fiducials[3]:fiducials[5]] = st

    contributions = 0
    bad_pr = np.where((ecg_rms[fiducials[0]:fiducials[2]] > ecg_ref_rms_max[fiducials[0]:fiducials[2]]) | (ecg_rms[fiducials[0]:fiducials[2]] < ecg_ref_rms_min[fiducials[0]:fiducials[2]]))[0]
    bad_qrs = np.where((ecg_rms[fiducials[2]:fiducials[3]] > ecg_ref_rms_max[fiducials[2]:fiducials[3]]) | (ecg_rms[fiducials[2]:fiducials[3]] < ecg_ref_rms_min[fiducials[2]:fiducials[3]]))[0]
    bad_st = np.where((ecg_rms[fiducials[3]:fiducials[5]] > ecg_ref_rms_max[fiducials[3]:fiducials[5]]) | (ecg_rms[fiducials[3]:fiducials[5]] < ecg_ref_rms_min[fiducials[3]:fiducials[5]]))[0]

    contributions = len(bad_pr) / len(ecg_rms[fiducials[0]:fiducials[2]]) * 0.5 + len(bad_qrs) / len(ecg_rms[fiducials[2]:fiducials[3]]) + len(bad_st) / len(ecg_rms[fiducials[3]:fiducials[5]]) * 1.5
    contributions *= 100/3

    saliency_map *= contributions / 100

    saliency_colors = cmap(saliency_map)

    fig, ax = plt.subplots(figsize=(4*resolution, 5*resolution))
    # fig.canvas.mpl_connect('resize_event', on_resize)
    x = np.arange(0,400,1)
    colors_lead = saliency_colors[:, :3]  # Extract the RGB colors for the lead
    # plt.fill_between(x, ecg_ref_rms_min, ecg_ref_rms_max, alpha=0.3, color='black')

    for j in range(400 - 1):
        if j <= start or j >= end:
            pass
            # ax.plot(x[j:j+2], ecg_rms[j:j+2], color=color0, linewidth=2*resolution, alpha=0.8)
        else:
            ax.plot(x[j:j+2], ecg_rms[j:j+2], color=colors_lead[j:j+1], linewidth=2*resolution)

    ax.plot(x[:start], np.repeat(ecg_rms[start],start), linewidth=2*resolution, color=color0)
    ax.plot(x[end:], np.repeat(ecg_rms[end],400-end), linewidth=2*resolution, color=color0)

    ax.text(x=10, y=2.3, s='RMS', fontsize=20*resolution, weight='bold')
    ax.text(x=10, y=2.1, s='{0:.0f}%'.format(contributions), fontsize=16*resolution, weight='bold')

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
    plt.ylim((0, 2.5))
    plt.title(id.split('/')[-1][:-4], fontsize=20)
    fig.tight_layout()
    plt.show()
    # plt.savefig('C:/Users/nater/OneDrive/Desktop/true positives diff norm/{}_rms.png'.format(id.split('/')[-1][:-4]), dpi=300)
    # plt.close()




print('done')