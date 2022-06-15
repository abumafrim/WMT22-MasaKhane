import os

%cd /content/drive/MyDrive/

if not os.path.exists('WMT22-MasaKhane/data/mmt-africa-format/mafand_mt'):
    print("Creating project and data folders...")
    os.makedirs("WMT22-MasaKhane/data/mmt-africa-format/mafand_mt")

%cd WMT22-MasaKhane

!git clone https://github.com/masakhane-io/lafand-mt.git

%cd lafand-mt/data/text_files

import pandas as pd

src_lang = 'en'
en_tgt = ['hau', 'ibo', 'lug', 'swa', 'tsn', 'yor', 'zul']
fr_tgt = ['wol']

d_types = ['dev', 'test', 'train']

for tgt_lang in en_tgt:
  for d_type in d_types:
    with open(src_lang + '_' + tgt_lang + '_news/' + d_type + '.' + src_lang, 'r') as f:
      src = f.readlines()

    with open(src_lang + '_' + tgt_lang + '_news/' + d_type + '.' + tgt_lang, 'r') as f:
      tgt = f.readlines()

    src = [x.strip() for x in src]
    tgt = [x.strip() for x in tgt]

    df = pd.DataFrame(list(zip(src, tgt)), columns=['input','target'])
    df.to_csv('../../../data/mmt-africa-format/mafand_mt/' + src_lang + '_' + tgt_lang + '_news_mafand_mt_' + d_type + '.tsv', sep='\t', index=False)

src_lang = 'fr'
for tgt_lang in fr_tgt:
  for d_type in d_types:
    with open(src_lang + '_' + tgt_lang + '_news/' + d_type + '.' + src_lang, 'r') as f:
      src = f.readlines()

    with open(src_lang + '_' + tgt_lang + '_news/' + d_type + '.' + tgt_lang, 'r') as f:
      tgt = f.readlines()

    src = [x.strip() for x in src]
    tgt = [x.strip() for x in tgt]

    df = pd.DataFrame(list(zip(src, tgt)), columns=['input','target'])
    df.to_csv('../../../data/mmt-africa-format/mafand_mt/' + src_lang + '_' + tgt_lang + '_news_mafand_mt_' + d_type + '.tsv', sep='\t', index=False)
