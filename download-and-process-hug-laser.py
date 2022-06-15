%cd /content/drive/MyDrive

if not os.path.exists('WMT22-MasaKhane/data/huggingface_raw'):
    print("Creating project and data folders...")
    os.makedirs("WMT22-MasaKhane/data/huggingface_raw")

if not os.path.exists('WMT22-MasaKhane/data/mmt-africa-format/huggingface_laser'):
    print("Creating project and data folders...")
    os.makedirs("WMT22-MasaKhane/data/mmt-africa-format/huggingface_laser")

%cd WMT22-MasaKhane/data/huggingface_raw

from sh import gunzip
import pandas as pd
import glob
import os

hug_langs = ['eng-hau', 'eng-ibo', 'eng-lug', 'eng-swh', 'eng-tsn', 'eng-yor', 'eng-zul', 'fra-wol']

for lang in hug_langs:
  url = 'https://huggingface.co/datasets/allenai/wmt22_african/resolve/main/wmt22_african_' + lang + '.gz'
  !wget {url}

  fname = 'wmt22_african_' + lang + '.gz'
  gunzip(fname)

  with open('wmt22_african_' + lang, 'r') as f:
    lines = f.readlines()

  src = []
  tgt = []

  for line in lines:
    src.append(line.split('\t')[0])
    tgt.append(line.split('\t')[1])

  df = pd.DataFrame(list(zip(src, tgt)), columns=['input','target'])
  df.to_csv('../mmt-africa-format/huggingface_laser/wmt22_african_' + lang + '-para.tsv', sep='\t', index=False)
