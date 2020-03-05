import os
import json
import copy

fname = 'cuckoo_pack1_unknown'
path = '../../Dataset/{}'.format(fname)
allow_cats = ['registry', 'file', 'process']

flags_data = []
flags_data_tags = []

flags_keys = ['desired_access',
'reg_type',
'information_class',
'protection',
'allocation_type',
'create_disposition',
'create_options',
'file_attributes',
'status_info',
'share_access',
'win32_protect',
'open_options',
'control_code',
'creation_flags']



'''
vocabs = []

for class_name in os.listdir(path):
    vocabs_class = []

    for json_file in os.listdir(os.path.join(path, class_name)):
        json_path = os.path.join(path, class_name, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)

            print(json_path)
            if 'behavior' not in data:
                print('\t No behavior tag. Skip!')
                continue
            for proc in data['behavior']['processes']:
                for api in proc['calls']:
                    if api['category'] in allow_cats:
                        flags = api['flags']

                        for flag_key in flags:
                            # print('flag_key', flag_key, ' || flags_keys_data_class[flag_key]', flags_keys_data_class[flag_key])
                            flag_values = flags[flag_key].split('|')

                            if flag_key not in vocabs:
                                vocabs.append(flag_key)
                            if flag_key not in vocabs_class:
                                vocabs_class.append(flag_key)

                            for flag_value in flag_values:
                                if flag_value not in vocabs:
                                    vocabs.append(flag_value)
                                if flag_value not in vocabs_class:
                                    vocabs_class.append(flag_value)

    with open('analyze/{}__vocabs__{}.txt'.format(fname, class_name), 'w') as f:
        f.write(' '.join(vocabs_class))


with open('analyze/{}__vocabs.txt'.format(fname), 'w') as f:
    f.write(' '.join(vocabs))
'''


# replace n with ' '
with open('analyze/{}__vocabs.txt'.format(fname), 'r+') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    f.seek(0)
    f.write(' '.join(lines))
    f.truncate()
with open('analyze/{}__vocabs__benign.txt'.format(fname), 'r+') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    print(' '.join(lines))
    f.seek(0)
    f.write(' '.join(lines))
    f.truncate()
with open('analyze/{}__vocabs__malware.txt'.format(fname), 'r+') as f:
    lines = f.readlines()
    lines = [line.strip() for line in lines]
    f.seek(0)
    f.write(' '.join(lines))
    f.truncate()
