data_path = '../other-repos/Web-Crawl-African/data/African_English_parallel_corpora/'
processed_path = 'processed/webcrawl_african/'
d_type = 'train'
langs = ['hau', 'ibo', 'lug', 'swh', 'tsn', 'yor', 'zul']

import os
import shutil

print('\nWebCrawl_African\n')

if not os.path.exists(processed_path):
    print('Creating {0}...'.format(processed_path))
    os.makedirs(processed_path)

for lang in langs:
    src = data_path + 'webcrawl-african-' + lang + '-eng.eng'
    tgt = data_path + 'webcrawl-african-' + lang + '-eng.' + lang
    src_dst = processed_path + d_type + '.eng-' + lang + '.eng'
    tgt_dst = processed_path + d_type + '.eng-' + lang + '.' + lang

    shutil.copyfile(src, src_dst)
    shutil.copyfile(tgt, tgt_dst)

    print('Finish eng-' + lang)