import os
import json
import gdown
import zipfile

raw_path = 'raw/lava-corpus/'
processed_path = 'processed/lava-corpus/'
d_type = 'train'
langs = []
pairs = []

print('\nLava\n')

if not os.path.exists(raw_path):
    print("Creating {0}...".format(raw_path))
    os.makedirs(raw_path)

if not os.path.exists(processed_path):
    print("Creating {0}...".format(processed_path))
    os.makedirs(processed_path)

url = 'https://drive.google.com/uc?id=1iNlEJuJWQZp5ZmKWMfO89lDwHolHty6_'
fname = raw_path + 'wmt22_lava_corpus_v1.zip'

try:
    gdown.download(url, fname, quiet=False)

    with zipfile.ZipFile(fname, 'r') as f:
        f.extractall(raw_path)

    with open(raw_path + 'wmt22_lava_corpus_v1.txt', 'r') as f:
        lines = f.readlines()

    for data in lines:
        lang = [json.loads(data)['src_lang'], json.loads(data)['trg_lang']]
        if lang not in langs:
            langs.append(lang)

    for lang in langs:
        pairs.append(lang[0] + '-' + lang[1])

    all_data = {item: [] for item in pairs}

    for data in lines:
        all_data[json.loads(data)['src_lang'] + '-' + json.loads(data)['trg_lang']].append([json.loads(data)['src_text'], json.loads(data)['trg_text']])

    for key, values in all_data.items():
        s_t = []
        t_t = []
        for value in values:
            s_t.append(value[0].strip() + '\n')
            t_t.append(value[1].strip() + '\n')

        with open(processed_path + d_type + '.' + key + '.' + key.split('-')[0], 'w') as f:
            f.writelines(s_t)

        with open(processed_path + d_type + '.' + key + '.' + key.split('-')[1], 'w') as f:
            f.writelines(t_t)

        print('Finished: {0}'.format(key))
except:
    print("File not found, please check the link at (https://statmt.org/wmt22/large-scale-multilingual-translation-task.html)")