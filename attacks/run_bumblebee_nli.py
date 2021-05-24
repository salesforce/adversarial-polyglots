from transformers import glue_processors as processors
import csv, os, argparse, ray, torch, time, json
from pathlib import Path
from bumblebee import BumblebeePairSequenceClassificationHF
from tqdm import tqdm
from ray.util import ActorPool
from math import ceil

parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", default=None, type=str, required=True, help="The input data directory, e.g., 'data/MNLI'.")
parser.add_argument("--model", "-m", type=str, required=True)
parser.add_argument("--output_dir", "-o", default=None, type=str, required=True, help="The output directory.")
parser.add_argument("--mm", action='store_true', required=False, help="Use Mismatch dev data.")
parser.add_argument("--split", '-s', default='test', type=str, required=False, help="Use the train, dev, or test data as the source.")
parser.add_argument("--beam", '-b', default=1, type=int, required=False, help="Beam size.")
parser.add_argument("--alpha", '-a', default=1, type=float, required=False, help="Alpha.")
parser.add_argument("--tgt_langs", '-t', default='fr,es,de,zh,el,bg,ru,tr,ar,vi,th,hi,sw,ur', type=str, required=False, help="Embedded languages.")
parser.add_argument("--transliterate", '-trl', action='store_true', required=False, help="Transliterate non-Latin scripts")
parser.add_argument("--unsup", '-u', action='store_true', required=False, help="use unsupervised translations.")
parser.add_argument("--gpu", '-g', default=0.25, type=float, required=False, help="GPU allocation per actor (% of one GPU). Total number of parallel actors is calculated using this value. Set to 0 to use CPU.")
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available() and args.gpu > 0
NUM_GPU_PER_ACTOR = args.gpu if USE_CUDA else 0
SRC_LANG = 'en'
TGT_LANGS = args.tgt_langs.split(',')
LABELS = ["contradiction", "entailment", "neutral"]

@ray.remote(num_gpus=NUM_GPU_PER_ACTOR)
class PolyglotActor(object):
    def __init__(self, model, src_lang, tgt_langs, reference_translations, transliteration_map=None):
        self.polyglot = BumblebeePairSequenceClassificationHF(model, src_lang, tgt_langs, LABELS,
                                                              is_nli=True, use_cuda=USE_CUDA,
                                                              transliteration_map=transliteration_map)
        self.reference_translations = reference_translations

    def mutate(self, batch, beam):
        score = 0
        early_terminate = args.split == 'train'
        results = []
        for example in tqdm(batch):
            prem_refs = self.reference_translations[0][example.text_a]
            hypo_refs = self.reference_translations[1][example.text_b]
            if example.label == 'contradictory':
                example.label = 'contradiction'
            refs = [prem_refs, hypo_refs]
            prem, hypo, text_label, lg_counts, _, \
            lowest_prem, lowest_hypo, lowest_text_label, lowest_lg_counts = self.polyglot.generate(example.text_a,
                                                                                           example.text_b,
                                                                                           example.label,
                                                                                           reference_translations=refs,
                                                                                           beam_size=beam,
                                                                                           early_terminate=early_terminate)
            if sum(lg_counts.values()) > 0:
                example.text_a = prem.replace(' ,', ',').replace(' .', '.')\
                                     .replace(" '", "'").replace('( ', '(')\
                                     .replace(' )', ')').replace(' ?', '?').replace(' !', '!')
                example.text_b = hypo.replace(' ,', ',').replace(' .', '.')\
                                     .replace(" '", "'").replace('( ', '(')\
                                     .replace(' )', ')').replace(' ?', '?').replace(' !', '!')

                example.text_a_lowest = lowest_prem.replace(' ,', ',').replace(' .', '.')\
                                                   .replace(" '", "'").replace('( ', '(')\
                                                   .replace(' )', ')').replace(' ?', '?').replace(' !', '!')
                example.text_b_lowest = lowest_hypo.replace(' ,', ',').replace(' .', '.')\
                                                   .replace(" '", "'").replace('( ', '(')\
                                                   .replace(' )', ')').replace(' ?', '?').replace(' !', '!')

                example.adv_label = text_label
                example.adv_label_lowest = lowest_text_label
                example.lg_counts = lg_counts
                if args.split == 'train':
                    results.append(example)
            elif text_label == example.label:
                score +=1

            if args.split != 'train': # in testing mode we want to output all examples, not just the successfully perturbed ones, for easy evaluation
                results.append(example)

        return results, score


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
            #output_line[4] = '-'
            #output_line[5] = '-'
            #output_line[6] = '-'
            #output_line[7] = '-'
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

def collate_lg_counts(examples, lgs):
    lg_counts = {lg:0 for lg in lgs}
    for example in examples:
        try:
            for lg, count in example.lg_counts.items():
                lg_counts[lg] += count
        except AttributeError:
            continue
    return lg_counts


def get_examples(data_dir, task, split):
    if split == 'dev':
        return processors[task]().get_dev_examples(data_dir)
    elif split == 'test':
        return processors[task]().get_test_examples(data_dir)
    elif split == 'train':
        return processors[task]().get_train_examples(data_dir)
    raise ValueError('Must be train, dev, or test')


if args.unsup:
    unsup_param = '.unsup'
else:
    unsup_param = ''

if args.transliterate:
    transliterate_param = '.translit'
else:
    transliterate_param = ''

output_path = Path(args.output_dir,
                   'bumblebee-pair_' + \
                   args.data.split('/')[-1] + '.' + args.split + \
                   '.' + args.model.strip('/').split('/')[-1] + \
                   '.' + '_'.join(TGT_LANGS) + unsup_param + transliterate_param + \
                   '.beam-' + str(args.beam) + '.equiv_constr')


output_file = args.split
if args.mm:
    output_file += '_mismatched.tsv'
    input_tsv = processors['mnli-mm']()._read_tsv(Path(args.data, output_file))
    examples = get_examples(args.data, 'mnli-mm', args.split)
else:
    if args.split != 'train':
        output_file += '_matched'
    output_file += '.tsv'
    input_tsv = processors['mnli']()._read_tsv(Path(args.data, output_file))
    examples = get_examples(args.data, 'mnli', args.split)

output_file = str(Path(output_path, output_file))
print('Output file path:', output_file)

if args.transliterate:
    transliteration_map = json.load(open('en-hi.transliterations.json','r'))
else:
    transliteration_map = None

if args.unsup:
    reference_translations = [json.load(open('../dictionaries/xnli-unsup-sentence1-reference-translations-en-head.json','r')),
                          json.load(open('../dictionaries/xnli-unsup-sentence2-reference-translations-en-head.json','r'))]
else:
    reference_translations = [json.load(open('../dictionaries/xnli-'+args.split+'-sentence1-reference-translations-en-head.json','r')),
                          json.load(open('../dictionaries/xnli-'+args.split+'-sentence2-reference-translations-en-head.json','r'))]


num_actors = int(torch.cuda.device_count() // NUM_GPU_PER_ACTOR) if USE_CUDA else 10
print('Number of Polyglots:', num_actors)

total_exs = len(examples)
print(total_exs)
len_per_batch = ceil(total_exs / num_actors)

batches = [examples[i:i+len_per_batch] for i in range(0, total_exs, len_per_batch)]

ray.init()
actors = ActorPool([PolyglotActor.remote(args.model, SRC_LANG, TGT_LANGS, reference_translations, transliteration_map)
                   for i in range(num_actors)])
start = time.time()
results, scores = map(list, zip(*actors.map(lambda actor, batch: actor.mutate.remote(batch, args.beam), batches)))
time_taken = time.time() - start
results = [ex for batch in results for ex in batch]

print("Acc:", str(sum(scores)/total_exs * 100))
print("Time taken:", time_taken / 60)
print("Output: ", str(output_path))
output_path.mkdir(parents=True, exist_ok=True)
_write_tsv(_create_output_data(results, input_tsv), output_file)

lg_counts = collate_lg_counts(results, TGT_LANGS)
json.dump(lg_counts, open(Path(output_path, 'lg_counts.json'),'w'))
print("Adversarial language distribution: ", str(lg_counts))
with open(str(Path(output_path, args.split+'_results.txt')), 'w') as t:
    t.write('Acc: '+str(sum(scores)/total_exs * 100)+'\n')
    t.write("Time taken: "+str(time_taken / 60))
