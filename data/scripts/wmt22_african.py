import os
import gzip
import requests

base_path = '../../'
raw_path = 'raw/wmt22_african/'
processed_path = 'processed/wmt22_african/'
d_type = 'train'
hug_langs = ['eng-hau', 'eng-ibo', 'eng-lug', 'eng-swh', 'eng-tsn', 'eng-yor', 'eng-zul', 'fra-wol']

print('\nWMT22_African\n')

if not os.path.exists(raw_path):
  print("Creating {0}...".format(raw_path))
  os.makedirs(raw_path)

if not os.path.exists(processed_path):
  print("Creating {0}...".format(processed_path))
  os.makedirs(processed_path)

os.chdir(raw_path)

for lang in hug_langs:

  src = []
  tgt = []
    
  url = 'https://huggingface.co/datasets/allenai/wmt22_african/resolve/main/wmt22_african_' + lang + '.gz'
  r = requests.get(url, allow_redirects=True)
  fname = raw_path + lang + '.gz'
  
  open(fname, 'wb').write(r.content)
  with gzip.open(fname, 'rb') as f:
    lines = f.readlines()

  for line in lines:
    src.append(line.split('\t')[0].strip() + '\n')
    tgt.append(line.split('\t')[1].strip() + '\n')

  with open(base_path + processed_path + d_type + '.' + lang + '.' + lang.split('-')[0], 'w') as f:
    f.writelines(src)

  with open(base_path + processed_path + d_type + '.' + lang + '.' + lang.split('-')[1], 'w') as f:
    f.writelines(tgt)

  print('Finished: ' + lang)

os.chdir(base_path)