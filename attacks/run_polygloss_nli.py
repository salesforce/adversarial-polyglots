from transformers import glue_processors as processors
import csv, os, argparse, ray, torch, time, json
from pathlib import Path
from polygloss import PolyglossPairSequenceClassificationHF
from tqdm import tqdm
from ray.util import ActorPool
from math import ceil

parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", default=None, type=str, required=True, help="The input data directory, e.g., 'data/MNLI'.")
parser.add_argument("--model", "-m", type=str, required=True)
parser.add_argument("--output_dir", "-o", default=None, type=str, required=True, help="The output directory.")
parser.add_argument("--mm", action='store_true', required=False, help="Use Mismatch dev data.")
parser.add_argument("--split", '-s', default='test', type=str, required=False, help="Use the dev or test data as the source.")
parser.add_argument("--beam", '-b', default=0, type=int, required=False, help="Beam size.")
parser.add_argument("--tgt_langs", '-t', default='zh,hi,fr,de,tr', type=str, required=False, help="Comma separated list of embedded languages.")
parser.add_argument("--use_reference_translations", '-r', action='store_true', required=False, help="Filter candidates with reference translations.")
parser.add_argument("--simplified_zh", '-szh', action='store_true', required=False, help="Use simplified Chinese dict")
parser.add_argument("--gpu", '-g', default=0.33, type=float, required=False, help="GPU allocation per actor (% of one GPU). Total number of parallel actors is calculated using this value. Set to 0 to use CPU.")

args = parser.parse_args()

USE_CUDA = torch.cuda.is_available() and args.gpu > 0
NUM_GPU_PER_ACTOR = args.gpu if USE_CUDA else 0 # set gpu usage
SRC_LANG = 'en' # matrix language
LABELS = ["contradiction", "entailment", "neutral"]

@ray.remote(num_gpus=NUM_GPU_PER_ACTOR)
class PolyglotActor(object):
    def __init__(self, model, src_lang, tgt_langs, src_tgts_map, reference_translations=None):
        self.polyglot = PolyglossPairSequenceClassificationHF(model, src_lang, tgt_langs, src_tgts_map, LABELS, is_nli=True, use_cuda=USE_CUDA)
        self.reference_translations = reference_translations

    def mutate(self, batch, beam):
        score = 0
        for example in tqdm(batch):
            prem_refs = self.reference_translations[0][example.text_a] if self.reference_translations else None
            hypo_refs = self.reference_translations[1][example.text_b] if self.reference_translations else None
            refs = [prem_refs, hypo_refs]
            prem, hypo, text_label, _, lowest_prem, lowest_hypo, lowest_text_label = self.polyglot.generate(example.text_a,
			                                                                                                example.text_b,
			                                                                                                example.label,
			                                                                                                beam_size=beam,
			                                                                                                reference_translations=refs)
            if text_label != example.label:
                example.text_a = prem
                example.text_b = hypo

                example.text_a_lowest = lowest_prem
                example.text_b_lowest = lowest_hypo

                example.adv_label = text_label
                example.adv_label_lowest = lowest_text_label
            else:
                score +=1
        return batch, score

def _create_output_data(examples, input_tsv_list):
    output = []
    columns = {}
    for (i, line) in enumerate(input_tsv_list):
            output_line = line.copy()
            if i == 0:
                output_line.insert(-1, 'predicted_label')
                output_line.insert(-1, 'sentence1_lowest')
                output_line.insert(-1, 'sentence2_lowest')
                output_line.insert(-1, 'predicted_label_lowest')
                columns = {col:i for i, col in enumerate(output_line)}
                output.append(output_line)
                continue
            output_line[4] = '-'
            output_line[5] = '-'
            output_line[6] = '-'
            output_line[7] = '-'
            output_line[columns['sentence1']] = examples[i-1].text_a
            output_line[columns['sentence2']] = examples[i-1].text_b
            try:
                output_line.insert(-1, examples[i-1].adv_label)
            except AttributeError:
                output_line.insert(-1, '-')
            try:
                output_line.insert(-1, examples[i-1].text_a_lowest)
                output_line.insert(-1, examples[i-1].text_b_lowest)
                output_line.insert(-1, examples[i-1].adv_label_lowest)
            except AttributeError:
                output_line.insert(-1, '-')
                output_line.insert(-1, '-')
                output_line.insert(-1, '-')
            output.append(output_line)
    return output

def _write_tsv(output, output_file):
    with open(output_file, "w", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter="\t", quotechar=None)
        for row in output:
            writer.writerow(row)

def get_examples(data_dir, task, split):
    if split == 'dev':
        return processors[task]().get_dev_examples(data_dir)
    if split == 'test':
        return processors[task]().get_test_examples(data_dir)
    raise ValueError('Must be dev or test')


TGT_LANGS = args.tgt_langs.split(',')

reference_translations = None
refs_var = 'no_ref'
if args.use_reference_translations:
    reference_translations = [json.load(open('../dictionaries/xnli-'+args.split+'-sentence1-reference-translations-en-head.json','r')),
                              json.load(open('../dictionaries/xnli-'+args.split+'-sentence2-reference-translations-en-head.json','r'))]
    refs_var = 'ref_constrained'

output_path = Path(args.output_dir,
                   'polygloss_pairseqcls_' + \
                   args.data.strip('/').split('/')[-1] + \
                   '.' + args.model.split('/')[-1] + \
                   '.' + '_'.join(TGT_LANGS) + \
                   '.beam-' + str(args.beam) + \
                    '.' + refs_var)
output_path.mkdir(parents=True, exist_ok=True)

output_file = args.split + '_'
if args.mm:
    output_file += 'mismatched'
else:
    output_file += 'matched'
output_file += '.tsv'
output_file = str(Path(output_path, output_file))
print('Output file path:', output_file)

if args.mm:
    input_tsv = processors['mnli-mm']()._read_tsv(args.data+'/'+args.split +'_mismatched.tsv')
    examples = get_examples(args.data, 'mnli-mm', args.split)
else:
    input_tsv = processors['mnli']()._read_tsv(args.data+'/'+args.split +'_matched.tsv')
    examples = get_examples(args.data, 'mnli', args.split)

if args.simplified_zh:
    word_map = 'en_to_all_map_simplified_zh.json'
else:
    word_map = 'en_to_all_map.json'


num_actors = int(torch.cuda.device_count() // NUM_GPU_PER_ACTOR) if USE_CUDA else int(25 // max(1, 0.5 * args.beam))
print('Number of Polyglots:', num_actors)

total_exs = len(examples)
print(total_exs)
len_per_batch = ceil(total_exs / num_actors)

batches = [examples[i:i+len_per_batch] for i in range(0, total_exs, len_per_batch)]

ray.init()
actors = ActorPool([PolyglotActor.remote(args.model, SRC_LANG, TGT_LANGS, word_map, reference_translations)
                   for i in range(num_actors)])
start = time.time()
results, scores = map(list, zip(*actors.map(lambda actor, batch: actor.mutate.remote(batch, args.beam), batches)))
time_taken = time.time() - start
results = [ex for batch in results for ex in batch]

print("Acc:", str(sum(scores)/total_exs * 100))
print("Time taken:", time_taken / 60)
print("Output: ", str(output_path))

_write_tsv(_create_output_data(results, input_tsv), output_file)
with open(str(Path(output_path, 'time_taken.txt')), 'w') as t:
    t.write('Acc: '+str(sum(scores)/total_exs * 100)+'\n')
    t.write("Time taken:"+str(time_taken / 60))
