import os

data_path = '../other-repos/lafand-mt/data/text_files/'
processed_path = 'processed/mafand_mt/'

print('\nMAFAND-MT\n')

if not os.path.exists(processed_path):
    print('Creating {0}...'.format(processed_path))
    os.makedirs(processed_path)

src_tgt = {'eng': ['hau', 'ibo', 'lug', 'swa', 'tsn', 'yor', 'zul'],'fra': ['wol']}
d_types = ['dev', 'test', 'train']

for src_lang, tgt in src_tgt.items():
  for tgt_lang in tgt:
    for d_type in d_types:
      with open(data_path + src_lang + '_' + tgt_lang + '_news/' + d_type + '.' + src_lang, 'r') as f:
        src = f.readlines()

      with open(data_path + src_lang + '_' + tgt_lang + '_news/' + d_type + '.' + tgt_lang, 'r') as f:
        tgt = f.readlines()

      src = [x.strip() for x in src]
      tgt = [x.strip() for x in tgt]

      with open(processed_path + d_type + '.' + src_lang + '-' + tgt_lang + '.' + src_lang, 'w') as f:
        f.writelines(src)

      with open(processed_path + d_type + '.' + src_lang + '-' + tgt_lang + '.' + tgt_lang, 'w') as f:
        f.writelines(tgt)

    print('Finished: ' + src_lang + '-' + tgt_lang)