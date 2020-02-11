import os
import shutil

src_dir = '/media/tunguyen/Devs/Dataset/cuckoo_old'
dst_dir = '/media/tunguyen/Devs/Dataset/cuckoo_ADung'
src_dir_none = '/media/tunguyen/Devs/Dataset/cuckoo_none_detect'
dst_dir_none = '/media/tunguyen/Devs/Dataset/cuckoo_ADung_none'

benigns = []
malwares = []
unknowns = []

with open('sha256_benign.txt', 'r') as f:
    lines = f.readlines()
    # benigns = [line.strip() for line in lines]
    for line in lines:
        filename = line.strip()
        shutil.copy(src_dir+'/benign/'+filename+'.json', dst_dir+'/benign/'+filename+'.json')

with open('sha256_malware.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        filename = line.strip()
        if os.path.exists(src_dir+'/malware/'+filename+'.json') and not os.path.exists(dst_dir+'/malware/'+filename+'.json'):
            shutil.copy(src_dir+'/malware/'+filename+'.json', dst_dir+'/malware/'+filename+'.json')
    
with open('sha256_unknown.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        filename = line.strip()
        if os.path.exists(src_dir_none+'/malware/'+filename+'.json') and not os.path.exists(dst_dir_none+'/malware/'+filename+'.json'):
            shutil.copy(src_dir_none+'/malware/'+filename+'.json', dst_dir_none+'/malware/'+filename+'.json')
    
