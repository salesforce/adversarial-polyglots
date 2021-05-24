import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", default=None, type=str, required=True, help="The input data file, e.g., 'data/XNLI/test.tsv'.")
parser.add_argument("--output_dir", "-o", default=None, type=str, required=True, help="The output directory.")
args = parse.parse_args()

LG = 'en'

out_file_path = os.path.join(args.output_dir, 'xnli-en', 'test_matched.tsv')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open(args.data, 'r') as fin, open(out_file_path, 'w') as fout:
    headerline = next(fin)
    fout.write(headerline)
    header = {col:i for i, col in enumerate(headerline.strip().split('\t'))}
    for i, line in enumerate(fin):
        cells = line.split('\t')
        if cells[header['language']] == LG:
            fout.write(line)
