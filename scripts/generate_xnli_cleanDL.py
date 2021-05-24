import json, random, argparse


parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", default=None, type=str, required=True, help="The input data file, e.g., 'data/XNLI/test.tsv'.")
parser.add_argument("--output_file", "-o", default=None, type=str, required=True, help="The output file path and name.")
args = parse.parse_args()


reference_translations = [json.load(open('../dictionaries/xnli-extended-sentence1-reference-translations-en_head.json','r')),
                          json.load(open('../dictionaries/xnli-extended-sentence1-reference-translations-en_head.json','r'))]


languages = ['en','fr','es','de','zh','el','bg','ru','tr','ar','vi','th','hi','sw','ur']

with open(args.data,'r') as fin, open(args.output_file,'w') as fout:
    headerline = next(fin).strip()
    fout.write(headerline+'\n')
    header = {col:i for i, col in enumerate(headerline.split('\t'))}
    for line in fin:
        cells = line.strip().split('\t')
        lg1 = random.choice(languages)
        lg2 = random.choice([lg for lg in languages if lg != lg1])
        new_sent1 = reference_translations[0][cells[header['sentence1']]][lg1]
        new_sent2 = reference_translations[1][cells[header['sentence2']]][lg2]
        cells[header['sentence1']] = new_sent1
        cells[header['sentence2']] = new_sent2
        fout.write('\t'.join(cells)+'\n')
