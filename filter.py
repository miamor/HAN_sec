import os
import shutil

with open('malware_report_A_Dung.txt', 'r') as f:
    for line in f:
        filename = line
        if os.path.exists('../Dataset/cuckoo/'+filename):
            shutil.move('../Dataset/cuckoo/'+filename, '../Dataset/cuckoo_split/'+filename)