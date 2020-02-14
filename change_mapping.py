from utils.utils import load_pickle
import torch

labels = torch.load('/media/tunguyen/Devs/Security/HAN_sec/data/adnew_iapi/pickle/labels')
labels_txt = load_pickle('/media/tunguyen/Devs/Security/HAN_sec/data/adnew_iapi/pickle/labels_txt')

labels_new = labels.clone()
for i,label in enumerate(labels):
    print(i, label)
    if label == 0: # current malware, change malware to 1
        labels_new[i] = 1
    
    if label == 1: # current benign, change benign to 0
        labels_new[i] = 0

torch.save(labels_new, '/media/tunguyen/Devs/Security/HAN_sec/data/adnew_iapi/pickle/labels_')