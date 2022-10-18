import os
import pandas as pd

hug_path = '../data/raw/wmt22_african/'
maf_path = '../data/processed/mafand_mt/'
langs = ['eng-hau', 'eng-ibo', 'eng-lug', 'eng-swh', 'eng-tsn', 'eng-yor', 'eng-zul', 'fra-wol']



print('\nFormatting auto-aligned data for prediction')

auto_aligned = ['wmt22_african', 'lava', 'webcrawl_african', 'WikiMatrix', 'CCAligned', 'CCMatrix', 'ParaCrawl', 'GNOME', 'KDE4', 'TED2020', 'XLEnt', 'Ubuntu', 'wikimedia', 'MultiCCAligned']

for data in auto_aligned:
  src_path = os.path.join('../data/processed/', data)
  files = os.listdir(src_path)
  langs = []

  for f in files:
    if f.startswith('train') and f[6:-4] not in langs:
      langs.append(f[6:-4])

  for lang in langs:
    with open(os.path.join(src_path, 'train.' + lang + '.' + lang.split('-')[0]), 'r') as f:
      src = f.readlines()
      src = [x.strip() for x in src]
    with open(os.path.join(src_path, 'train.' + lang + '.' + lang.split('-')[1]), 'r') as f:
      tgt = f.readlines()
      tgt = [x.strip() for x in tgt]

    tgt_path = os.path.join('../data/filtering', data)
    if not os.path.exists(tgt_path):
      print("Creating {0}".format(tgt_path))
      os.makedirs(tgt_path)

    df = pd.DataFrame({'sentence1': src, 'sentence2': tgt})
    df.to_csv(os.path.join(tgt_path, lang + '.tsv'), sep='\t', index=False)
    print('Finished: {0}'.format(lang))