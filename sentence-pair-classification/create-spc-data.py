import os
import pandas as pd

hug_path = '../data/raw/wmt22_african/'
maf_path = '../data/processed/mafand_mt/'
langs = ['eng-hau', 'eng-ibo', 'eng-lug', 'eng-swh', 'eng-tsn', 'eng-yor', 'eng-zul', 'fra-wol']

data_columns = ['sentence1', 'sentence2']

for lang in langs:

  path = 'data/' + lang
  if not os.path.exists(path):
    print("Creating {0}".format(lang))
    os.makedirs(path)

  #read mafand as positive examples
  with open(maf_path + 'train.' + lang + lang.split('-')[0], 'r') as f:
    src = f.readlines()
  with open(maf_path + 'train.' + lang + lang.split('-')[0], 'r') as f:
    tgt = f.readlines()
  train_df = pd.DataFrame({'sentence1': src, 'sentence2': tgt, 'label': [1] * len(src)})

  with open(maf_path + 'test.' + lang + lang.split('-')[0], 'r') as f:
    src = f.readlines()
  with open(maf_path + 'test.' + lang + lang.split('-')[0], 'r') as f:
    tgt = f.readlines()
  test_df = pd.DataFrame({'sentence1': src, 'sentence2': tgt, 'label': [1] * len(src)})

  with open(maf_path + 'dev.' + lang + lang.split('-')[0], 'r') as f:
    src = f.readlines()
  with open(maf_path + 'dev.' + lang + lang.split('-')[0], 'r') as f:
    tgt = f.readlines()
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
