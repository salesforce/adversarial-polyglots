import argparse, json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", default=None, type=str, required=True, help="The input data file, e.g., 'data/xnli.tsv'.")
parser.add_argument("--matrix", "-m", default='en', type=str, required=True, help="The matrix language.")
parser.add_argument("--output", "-o", default='../dictionaries', type=str, required=True, help="The output directory.")
parser.add_argument("--split", '-s', default='test', type=str, required=False, help="train, dev, or test.")


sentence1s = {}
sentence2s = {}

with open(args.data,'r') as fin:
    headerline = next(fin).strip()
    header = {col:i for i, col in enumerate(headerline.split('\t'))}
    for i, line in enumerate(fin):
        cells = line.split('\t')
        if not sentence1s.get(cells[header['pairID']]):
            sentence1s[cells[header['pairID']]] = {}
		if not sentence2s.get(cells[header['pairID']]):
            sentence2s[cells[header['pairID']]] = {}
        if cells[header['language']] == 'th' or cells[header['language']] == 'zh':
            sentence1s[cells[header['pairID']]][cells[header['language']]] = cells[header['sentence1_tokenized']]
			sentence2s[cells[header['pairID']]][cells[header['language']]] = cells[header['sentence2_tokenized']]
        else:
            sentence1s[cells[header['pairID']]][cells[header['language']]] = cells[header['sentence1']]
			sentence2s[cells[header['pairID']]][cells[header['language']]] = cells[header['sentence2']]

new_sentence1s = {}
new_sentence2s = {}

for d in (new_sentence1s, new_sentence2s):
	for k,v in d.items():
	    d[v[args.matrix]] = v

json.dump(new_sentence1s, open(Path(args.output,'xnli-'+args.split+'-sentence1-reference-translations-'+args.matrix+'-head.json'),'w'), indent=False, ensure_ascii=False)
json.dump(new_sentence2s, open(Path(args.output,'xnli-'+args.split+'-sentence2-reference-translations-'+args.matrix+'-head.json'),'w'), indent=False, ensure_ascii=False)
