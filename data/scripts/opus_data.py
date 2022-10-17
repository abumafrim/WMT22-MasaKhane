import os
import zipfile
import requests

print('\nOPUS datasets\n')

data = {'en':
            {
                'Tanzil': ['v1', 'moses', ['ha', 'sw']],
                'QED': ['v2.0a', 'moses', ['hau', 'swa', 'ibo', 'lg', 'yor', 'zul']],
                'WikiMatrix': ['v1', 'moses', ['sw']],
                'CCAligned': ['v1', 'moses', ['zu', 'ha', 'ig', 'lg', 'sw', 'tn', 'yo', 'zu']],
                'CCMatrix': ['v1', 'moses', ['ha', 'ig', 'sw']],
                'XLEnt': ['v1.1', 'moses', ['zu', 'ha', 'ig', 'lg', 'sw', 'tn', 'yo', 'zu']],
                'GlobalVoices': ['v2018q4', 'moses', ['sw', 'yo']],
                'ParaCrawl': ['v8', 'moses', ['sw']],
                'GNOME': ['v1', 'moses', ['zu', 'ha', 'ig', 'lg', 'sw', 'yo', 'zu']],
                'tico-19': ['v2020-10-28', 'moses', ['zu', 'ha', 'lg', 'sw', 'zu']],
                'ELRC_2922': ['v1', 'moses', ['sw']],
                'EUbookshop': ['v2', 'moses', ['sw']],
                'KDE4': ['v2', 'moses', ['ha']],
                'TED2020': ['v1', 'moses', ['ha', 'ig', 'sw']],
                'Tatoeba': ['v2022-03-03', 'moses', ['swh', 'ha', 'zu', 'ig', 'lg', 'tn', 'yo', 'zu']],
                'Ubuntu': ['v14.10', 'moses', ['zu', 'ha', 'ig', 'lg', 'sw', 'yo', 'zu']],
                'bible-uedin': ['v1', 'moses', ['zu']],
                'wikimedia': ['v20210402', 'moses', ['zu', 'ha', 'ig', 'lg', 'sw', 'tn', 'yo', 'zu']],
            },
        'fr': 
            {
                'XLEnt': ['v1.1', 'moses', ['wo']],
                'Tatoeba': ['v2022-03-03', 'moses', ['wo']],
                'Ubuntu': ['v14.10', 'moses', ['wo']],
                'bible-uedin': ['v1', 'moses', ['wo']],
                'wikimedia': ['v20210402', 'moses', ['wo']],
                'QED': ['v2.0a', 'moses', ['wol']],
                'MultiCCAligned': ['v1.1', 'moses', ['wo']]
            }
        }

lang_code_map = {
                    'en': 'eng',
                    'fr': 'fra',
                    'ha': 'hau',
                    'hau': 'hau',
                    'ig': 'ibo',
                    'ibo': 'ibo',
                    'lg': 'lug',
                    'lug': 'lug',
                    'sw': 'swh',
                    'swh': 'swh',
                    'swa': 'swh',
                    'tn': 'tsn',
                    'tsn': 'tsn',
                    'yo': 'yor',
                    'yor': 'yor',
                    'wo': 'wol',
                    'wol': 'wol',
                    'zu': 'zul',
                    'zul': 'zul'
                }

d_type = 'train'

for src_lang, sources in data.items():

    for index, details in sources.items():

        raw_path = 'raw/' + index + '/'
        processed_path = 'processed/' + index + '/'

        if not os.path.exists(raw_path):
            print("Creating {0}...".format(raw_path))
            os.makedirs(raw_path)

        if not os.path.exists(processed_path):
            print("Creating {0}...".format(processed_path))
            os.makedirs(processed_path)

        for tgt_lang in details[2]:
            
            url = 'https://object.pouta.csc.fi/OPUS-' + index + '/' + details[0] + '/' + details[1] + '/' + src_lang + '-' + tgt_lang + '.txt.zip'
            
            r = requests.get(url, allow_redirects=True)
            fname = raw_path + src_lang + '-' + tgt_lang + '.txt.zip'
            open(fname, 'wb').write(r.content)

            with zipfile.ZipFile(fname, 'r') as f:
                f.extractall(raw_path)

            with open(raw_path + index + '.' + src_lang + '-' + tgt_lang + '.' + src_lang) as f:
                src_lines = f.readlines()

            with open(raw_path + index + '.' + src_lang + '-' + tgt_lang + '.' + tgt_lang) as f:
                tgt_lines = f.readlines()

            src_lines = [x.strip() for x in src_lines]
            tgt_lines = [x.strip() for x in tgt_lines]

            with open(processed_path + d_type + '.' + lang_code_map.get(src_lang) + '-' + lang_code_map.get(tgt_lang) + '.' + lang_code_map.get(src_lang), 'w') as f:
                f.writelines(src_lines)

            with open(processed_path + d_type + '.' + lang_code_map.get(src_lang) + '-' + lang_code_map.get(tgt_lang) + '.' + lang_code_map.get(tgt_lang), 'w') as f:
                f.writelines(tgt_lines)