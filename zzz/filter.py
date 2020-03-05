import os
import shutil

path = '/media/tunguyen/Devs/Dataset/cuckoo/malware'
with open('malware_report_A_Dung.txt', 'r') as f:
    lines = [line.rstrip() for line in f]

    for filename in lines:
        if os.path.exists(path+'/'+filename):
            shutil.move(path+'/'+filename, '/media/tunguyen/Devs/Dataset/cuckoo_split/classified/malware/'+filename)