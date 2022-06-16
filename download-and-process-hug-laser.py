from sh import gunzip
import pandas as pd
import glob
import os
import requests

if not os.path.exists('data/huggingface_raw'):
    print("Creating project and data folders...")
    os.makedirs("data/huggingface_raw")

if not os.path.exists('data/mmt-africa-format/huggingface_laser'):
    print("Creating project and data folders...")
    os.makedirs("data/mmt-africa-format/huggingface_laser")

hug_langs = ['eng-hau', 'eng-ibo', 'eng-lug', 'eng-swh', 'eng-tsn', 'eng-yor', 'eng-zul', 'fra-wol']

for lang in hug_langs:
    
  url = 'https://huggingface.co/datasets/allenai/wmt22_african/resolve/main/wmt22_african_' + lang + '.gz'
  r = requests.get(url, allow_redirects=True)
  fname = 'data/huggingface_raw/wmt22_african_' + lang + '.gz'  
  
  open(fname, 'wb').write(r.content)
  gunzip(fname)

  with open('data/huggingface_raw/wmt22_african_' + lang, 'r') as f:
    lines = f.readlines()

  src = []
  tgt = []

  for line in lines:
    src.append(line.split('\t')[0])
    tgt.append(line.split('\t')[1])

  df = pd.DataFrame(list(zip(src, tgt)), columns=['input','target'])
  df.to_csv('data/mmt-africa-format/huggingface_laser/wmt22_african_' + lang + '-para.tsv', sep='\t', index=False)

  print('Finished: ' + lang)