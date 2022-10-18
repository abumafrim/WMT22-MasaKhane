import os
import pandas as pd

hug_path = '../data/raw/wmt22_african/'
maf_path = '../data/processed/mafand_mt/'
langs = ['eng-hau', 'eng-ibo', 'eng-lug', 'eng-swh', 'eng-tsn', 'eng-yor', 'eng-zul', 'fra-wol']

data_columns = ['sentence1', 'sentence2']

print('Creating training data')

for lang in langs:

  path = 'data/' + lang
  if not os.path.exists(path):
    print("Creating {0}".format(lang))
    os.makedirs(path)

  #read mafand as positive examples
  with open(maf_path + 'train.' + lang + '.' + lang.split('-')[0], 'r') as f:
    src = f.readlines()
    src = [x.strip() for x in src]
  with open(maf_path + 'train.' + lang + '.' + lang.split('-')[1], 'r') as f:
    tgt = f.readlines()
    tgt = [x.strip() for x in tgt]
  train_df = pd.DataFrame({'sentence1': src, 'sentence2': tgt, 'label': [1] * len(src)})

  with open(maf_path + 'test.' + lang + '.' + lang.split('-')[0], 'r') as f:
    src = f.readlines()
    src = [x.strip() for x in src]
  with open(maf_path + 'test.' + lang + '.' + lang.split('-')[1], 'r') as f:
    tgt = f.readlines()
    tgt = [x.strip() for x in tgt]
  test_df = pd.DataFrame({'sentence1': src, 'sentence2': tgt, 'label': [1] * len(src)})

  with open(maf_path + 'dev.' + lang + '.' + lang.split('-')[0], 'r') as f:
    src = f.readlines()
    src = [x.strip() for x in src]
  with open(maf_path + 'dev.' + lang + '.' + lang.split('-')[1], 'r') as f:
    tgt = f.readlines()
    tgt = [x.strip() for x in tgt]
  dev_df = pd.DataFrame({'sentence1': src, 'sentence2': tgt, 'label': [1] * len(src)})
  
  #sample huggingface worst laser scores as negative examples
  with open(hug_path + lang, 'r') as f:
    lines = f.readlines()

  src = []
  tgt = []
  laser_score = []

  for line in lines:
    src.append(line.split('\t')[0])
    tgt.append(line.split('\t')[1])
    laser_score.append(line.split('\t')[2].rstrip().split()[0])

  hug_df = pd.DataFrame(list(zip(src, tgt, laser_score)), columns=[lang.split('-')[0], lang.split('-')[1], 'lscore'])
  
  hug_df = hug_df.sort_values(by=['lscore']).iloc[:, 0:2]
  hug_df.columns = ['sentence1', 'sentence2']

  hug_df = hug_df.assign(label = [0] * len(hug_df))

  #select from huggingface the same number of sentences as mafand and mix
  train_df = train_df.append(hug_df[:len(train_df)])
  dev_df = dev_df.append(hug_df[len(train_df):len(train_df) + len(dev_df)])
  test_df = test_df.append(hug_df[len(train_df) + len(dev_df):len(train_df) + len(dev_df) + len(test_df)])

  #save training and evaluation datasets
  train_df.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(path, 'spc-' + lang + '_train.tsv'), sep='\t', index=False)
  dev_df.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(path, 'spc-' + lang + '_dev.tsv'), sep='\t', index=False)
  test_df.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(path, 'spc-' + lang + '_test.tsv'), sep='\t', index=False)

  print('Finished: ' + lang)

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
    with open(src_path + 'train.' + lang + '.' + lang.split('-')[0], 'r') as f:
      src = f.readlines()
      src = [x.strip() for x in src]
    with open(src_path + 'train.' + lang + '.' + lang.split('-')[1], 'r') as f:
      tgt = f.readlines()
      tgt = [x.strip() for x in tgt]

    tgt_path = os.path.join('../data/filtering', data)
    if not os.path.exists(tgt_path):
      print("Creating {0}".format(tgt_path))
      os.makedirs(tgt_path)

    df = pd.DataFrame({'sentence1': src, 'sentence2': tgt})
    df.to_csv(os.path.join(tgt_path, lang + '.tsv'), sep='\t', index=False)