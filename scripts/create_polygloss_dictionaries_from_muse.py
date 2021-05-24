import json
from os import listdir
from os.path import isfile, join
from opencc import OpenCC
cc = OpenCC('t2s')

'''
Output format
{
    "word": {
        "lg": [word, word, word],
        "lg": [word, word, word],
    }
}
'''

# get all files in dictionaries and lg ids, each dictionary should be named en-<lg>.txt
dict_path = './dictionaries'
dict_files = {f.split('.')[0].split('-')[1]:f for f in listdir(dict_path) if isfile(join(dict_path, f))}
print(dict_files)

en_to_all_map = {}

# populate en_to_all_map, use simplified zh
for lg_id, file in dict_files.items():
    with open(join(dict_path, file),'r') as fin:
        for line in fin:
            if '\t' in line:
                en_word, other_word = line.strip().split('\t')
            else:
                en_word, other_word = line.strip().split()

            if en_word == other_word:
                continue
            print(en_word, other_word)
            if not en_to_all_map.get(en_word):
                en_to_all_map[en_word] = {}
            if not en_to_all_map[en_word].get(lg_id):
                en_to_all_map[en_word][lg_id] = []
            if lg_id == 'zh':
                other_word = cc.convert(other_word)
                print(other_word)
            en_to_all_map[en_word][lg_id].append(other_word)



json.dump(en_to_all_map,open('en_to_all_map_simplified_zh.json','w'),indent=4,ensure_ascii=False)
