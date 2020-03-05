import os
import shutil

# fid = '346'
# ftype = 'unknown'

# src_dir = '/media/fitmta/Storage/MinhTu/Dataset/Pack2'
# dst_dir = '/media/fitmta/Storage/MinhTu/Dataset/cuckoo_pack2_unknown'

# with open(src_dir+'/hash--ClamAVout_name_list_report_'+fid+'_virus_dir.txt', 'r') as f:
#     files = f.readlines()

#     for gname_dir in files:
#         filepath = gname_dir.strip()

#         src_path = src_dir+'/'+ftype+'_'+fid+'/'+filepath
#         dst_path = dst_dir+'/malware/'+filepath

#         if os.path.exists(src_path):
#             print('Existed! Moved')
#             shutil.move(src_path, dst_path)


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
