import os

if not os.path.exists('data/mmt-africa-format/mafand_mt'):
    print("Creating project and data folders...")
    os.makedirs("data/mmt-africa-format/mafand_mt")

path = 'lafand-mt/data/text_files/'

import pandas as pd

src_lang = 'en'
en_tgt = ['hau', 'ibo', 'lug', 'swa', 'tsn', 'yor', 'zul']
fr_tgt = ['wol']

d_types = ['dev', 'test', 'train']

for tgt_lang in en_tgt:
  for d_type in d_types:
    with open(path + src_lang + '_' + tgt_lang + '_news/' + d_type + '.' + src_lang, 'r') as f:
      src = f.readlines()

    with open(path + src_lang + '_' + tgt_lang + '_news/' + d_type + '.' + tgt_lang, 'r') as f:
      tgt = f.readlines()

    src = [x.strip() for x in src]
    tgt = [x.strip() for x in tgt]

    df = pd.DataFrame(list(zip(src, tgt)), columns=['input','target'])
    df.to_csv('data/mmt-africa-format/mafand_mt/' + src_lang + '_' + tgt_lang + '_news_mafand_mt_' + d_type + '.tsv', sep='\t', index=False)

  print('Finished: ' + src_lang + '_' + tgt_lang)

src_lang = 'fr'
for tgt_lang in fr_tgt:
  for d_type in d_types:
    with open(path + src_lang + '_' + tgt_lang + '_news/' + d_type + '.' + src_lang, 'r') as f:
      src = f.readlines()

    with open(path + src_lang + '_' + tgt_lang + '_news/' + d_type + '.' + tgt_lang, 'r') as f:
      tgt = f.readlines()

    src = [x.strip() for x in src]
    tgt = [x.strip() for x in tgt]

    df = pd.DataFrame(list(zip(src, tgt)), columns=['input','target'])
    df.to_csv('data/mmt-africa-format/mafand_mt/' + src_lang + '_' + tgt_lang + '_news_mafand_mt_' + d_type + '.tsv', sep='\t', index=False)
  
  print('Finished: ' + src_lang + '_' + tgt_lang)