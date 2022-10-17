import os
import gzip
import shutil
import requests
import pandas as pd

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

for lang in hug_langs:

  src = []
  tgt = []
    
  url = 'https://huggingface.co/datasets/allenai/wmt22_african/resolve/main/wmt22_african_' + lang + '.gz'
  r = requests.get(url, allow_redirects=True)
  fname = raw_path + lang
  
  open(fname, 'wb').write(r.content)
  with gzip.open(fname + '.gz', 'rb') as f_in:
    with open(fname, 'wb') as f_out:
      shutil.copyfileobj(f_in, f_out)

  df = pd.read_csv(fname, sep='\t', header=None, dtype=str)

  src = list(df.columns[0])
  src = [x.strip() + '\n' for x in src]
  tgt = list(df.columns[1])
  tgt = [x.strip() + '\n' for x in tgt]

  with open(processed_path + d_type + '.' + lang + '.' + lang.split('-')[0], 'w') as f:
    f.writelines(src)

  with open(processed_path + d_type + '.' + lang + '.' + lang.split('-')[1], 'w') as f:
    f.writelines(tgt)

  print('Finished: ' + lang)