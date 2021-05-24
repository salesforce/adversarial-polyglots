import argparse, json, jieba
from pythainlp.tokenize import word_tokenize
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", "-d", default=None, type=str, required=True, help="The input data directory, e.g., 'data'. Files are expected to be named 'xquad.<lg>.json'")
parser.add_argument("--matrix", "-m", default='en', type=str, required=True, help="The matrix language.")
parser.add_argument("--output", "-o", default='../dictionaries', type=str, required=True, help="The output directory.")

# Extract question


data = {}

for lg in ['en', 'es', 'de', 'el', 'ru', 'tr', 'ar', 'vi', 'th', 'zh', 'hi']:
    xquad_lg_data = json.load(open(Path(args.data_dir, 'xquad.'+lg+'.json'), 'r'))
    for article in xquad_lg_data['data']:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                question = qa['question']
                if lg == 'zh':
                    question = ' '.join(jieba.cut(qa['question'], cut_all=False))

                if lg == 'th':
                    question = ' '.join(word_tokenize(qa['question']))

				if not data.get(qa['id']):
		            data[qa['id']] = {}
                data[qa['id']][lg] = question

new_data = {}
for k,v in data.items():
    new_data[v[args.matrix]] = v

json.dump(new_data, open(Path(args.output,'xquad-question-reference-translations-'+args.matrix+'-head.json'),'w'), indent=False, ensure_ascii=False)
