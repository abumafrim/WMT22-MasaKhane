import glob
import os
import pandas as pd

hug_path = 'data/huggingface_raw/wmt22_african_'
hug_langs = ['eng-hau', 'eng-ibo', 'eng-lug', 'eng-swh', 'eng-tsn', 'eng-yor', 'eng-zul', 'fra-wol']

maf_path = 'data/mmt-africa-format/mafand_mt/'
maf_langs = ['en_hau', 'en_ibo', 'en_lug', 'en_swa', 'en_tsn', 'en_yor', 'en_zul', 'fr_wol']

for maf_lang, hug_lang in zip(maf_langs, hug_langs):

  path = 'sentence-pair-classification/data/' + hug_lang
  if not os.path.exists(path):
    print("Creation of the " + hug_lang + " folder...")
    os.system("mkdir " + path)

  #read mafand as positive examples
  train_df = pd.read_csv(maf_path + maf_lang + '_news_mafand_mt_train.tsv', sep='\t', header=None)
  dev_df = pd.read_csv(maf_path + maf_lang + '_news_mafand_mt_dev.tsv', sep='\t', header=None)
  test_df = pd.read_csv(maf_path + maf_lang + '_news_mafand_mt_test.tsv', sep='\t', header=None)

  train_df.columns = ['sentence1', 'sentence2']
  dev_df.columns = ['sentence1', 'sentence2']
  test_df.columns = ['sentence1', 'sentence2']

  train_df = train_df.assign(label = [1] * len(train_df))
  dev_df = dev_df.assign(label = [1] * len(dev_df))
  test_df = test_df.assign(label = [1] * len(test_df))
  
  #sample huggingface worst laser scores as negative examples
  with open(hug_path + hug_lang, 'r') as f:
    lines = f.readlines()

  src = []
  tgt = []
  laser_score = []

  for line in lines:
    src.append(line.split('\t')[0])
    tgt.append(line.split('\t')[1])
    laser_score.append(line.split('\t')[2].rstrip().split()[0])

  hug_df = pd.DataFrame(list(zip(src, tgt, laser_score)), columns=[hug_lang.split('-')[0], hug_lang.split('-')[1], 'laser_score'])
  
  hug_df = hug_df.sort_values(by=['laser_score']).iloc[:, 0:2]
  hug_df.columns = ['sentence1', 'sentence2']
  hug_df.to_csv(os.path.join(path, 'spc-' + maf_lang + '_to_classify.tsv'), sep='\t', index=False)

  hug_df = hug_df.assign(label = [0] * len(hug_df))

  #select from huggingface the same number of sentences as mafand and mix
  train_df = train_df.append(hug_df[:len(train_df)])
  dev_df = dev_df.append(hug_df[len(train_df):len(train_df) + len(dev_df)])
  test_df = test_df.append(hug_df[len(train_df) + len(dev_df):len(train_df) + len(dev_df) + len(test_df)])

  #save training and evaluation datasets
  train_df.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(path, 'spc-' + maf_lang + '_train.tsv'), sep='\t', index=False)
  dev_df.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(path, 'spc-' + maf_lang + '_dev.tsv'), sep='\t', index=False)
  test_df.sample(frac=1).reset_index(drop=True).to_csv(os.path.join(path, 'spc-' + maf_lang + '_test.tsv'), sep='\t', index=False)

  print('Finished: ' + hug_lang)
