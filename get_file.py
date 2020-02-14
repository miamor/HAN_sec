import os
import shutil

src_dir = '/media/fitmta/Storage/MinhTu/Dataset/cuckoo_old'
dst_dir = '/media/fitmta/Storage/MinhTu/Dataset/cuckoo_ADung'

with open('test_list.txt', 'r') as f:
    files = f.readlines()

    for gname_dir in files:
        filepath = gname_dir.strip().split('__')[1]
        folder = gname_dir.strip().split('__')[0]

        src_path = src_dir+'/'+folder+'/'+filepath
        dst_path = dst_dir+'/'+folder+'/'+filepath

        shutil.move(src_path, dst_path)
