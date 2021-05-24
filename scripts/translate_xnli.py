from transformers import MarianMTModel, MarianTokenizer
import json, argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", default=None, type=str, required=True, help="The input data file, e.g., 'data/XNLI/test.tsv'.")
parser.add_argument("--output_dir", "-o", default=None, type=str, required=True, help="The output directory.")
parser.add_argument("--include_header", "-o", default=True, type=bool, help="Whether to include header in the output tsv. Don't include if you want to concatenate it with the original tsv.")
args = parse.parse_args()


translator_paths = json.load(open('language-opus-nmt-map.json'))
translators = {lg: {"tokenizer": MarianTokenizer.from_pretrained(path),
                    "model": MarianMTModel.from_pretrained(path)}
               for lg, path in translator_paths.items() if path and lg in {'en','fr','es','de','zh','el','bg','ru','tr','ar','vi','th','hi','ur','sw'}}

print('Translators loaded.')


def translate(sentence, tokenizer, model):
    inputs = tokenizer.prepare_seq2seq_batch([sentence])
    outputs = model.generate(inputs['input_ids'], num_beams=5, early_stopping=True,num_return_sequences=1)
    return [tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False) for output in outputs]


out_file_path = os.path.join(args.output_dir, 'xnli-opus-hf', 'test_matched.tsv')

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

with open(args.data, 'r') as fin, open(out_file_path, 'w') as fout:
    headerline = next(fin).strip()
    header = {col:i for i, col in enumerate(headerline.split('\t'))}
	if args.include_header:
		fout.write(headerline+'\n')
    for i, line in enumerate(tqdm(fin, total=5010)):
        en_cells = line.split('\t')
        for lg in tqdm(translators.keys()):
            if lg in {'en','fr','es','de','zh','el','bg','ru','tr','ar','vi','th','hi','ur','sw'}:
                continue
            new_cells = en_cells.copy()
            tokenizer = translators[lg]['tokenizer']
            model = translators[lg]['model']
            new_cells[header['language']] = lg
            new_cells[header['sentence1_tokenized']] = ''
            new_cells[header['sentence2_tokenized']] = ''
            new_cells[header['sentence1']] = translate(en_cells[header['sentence1']], tokenizer, model)
            new_cells[header['sentence2']] = translate(en_cells[header['sentence2']], tokenizer, model)
            fout.write('\t'.join(new_cells))
