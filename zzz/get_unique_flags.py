import os
import json
import copy

fname = 'cuckoo_ADung_none'
path = '../../../Dataset/{}'.format(fname)
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


flags_keys_data = {}
for k in flags_keys:
    flags_keys_data[k] = {}

for class_name in os.listdir(path):
    flags_data_class = []
    flags_data_tags_class = []
    # flags_keys_data_class = copy.deepcopy(flags_keys_data)
    flags_keys_data_class = {}
    for k in flags_keys:
        flags_keys_data_class[k] = {}
    # print('flags_keys_data', flags_keys_data)

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

                            for flag_value in flag_values:
                                if flag_value not in flags_keys_data[flag_key]:
                                    # flags_keys_data[flag_key].append(flag_value)
                                    flags_keys_data[flag_key][flag_value] = 0
                                else:
                                    flags_keys_data[flag_key][flag_value] += 1
                                if flag_value not in flags_keys_data_class[flag_key]:
                                    # flags_keys_data_class[flag_key].append(flag_value)
                                    flags_keys_data_class[flag_key][flag_value] = 0
                                else:
                                    flags_keys_data_class[flag_key][flag_value] += 1

                                if flag_key not in flags_data_tags:
                                    flags_data_tags.append(flag_key)
                                if flag_key not in flags_data_tags_class:
                                    flags_data_tags_class.append(flag_key)

                                # flag_data = "{} : {}".format(flag_key, flag_value)

                                # if flag_data not in flags_data:
                                #     flags_data.append(flag_data)
                                # if flag_data not in flags_data_class:
                                #     flags_data_class.append(flag_data)
                                
                            # flag_data = '\n'.join("{}".format(val) for (val) in flags_keys_data[flag_key])

    with open('analyze/{}__flags_data_tags__{}.txt'.format(fname, class_name), 'w') as f:
        f.write('\n'.join(sorted(flags_data_tags_class)))


    flags_keys_data_class = {key: {k: flags_keys_data_class[k] for k in sorted(flags_keys_data_class)} for key in sorted(flags_keys)}
    with open('analyze/{}__flags_data__{}.json'.format(fname, class_name), 'w') as f:
        json.dump(flags_keys_data_class, f)

with open('analyze/{}__flags_data_tags.txt'.format(fname), 'w') as f:
    f.write('\n'.join(sorted(flags_data_tags)))

# flags_keys_data = {k: flags_keys_data[k] for k in sorted(flags_keys_data)}
flags_keys_data = {key: {k: flags_keys_data[k] for k in sorted(flags_keys_data)} for key in sorted(flags_keys)}
with open('analyze/{}__flags_data.json'.format(fname), 'w') as f:
    json.dump(flags_keys_data, f)

