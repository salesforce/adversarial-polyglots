from transformers import glue_processors as processors
import csv, os, argparse, json, ray, time, torch, jsonlines, random
from tqdm import tqdm
from pathlib import Path
from codemixer import CodeMixer
from ray.util import ActorPool
from math import ceil

parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", default=None, type=str, required=False, help="The input data file, e.g., 'data/MNLI/train.tsv'.")
parser.add_argument("--output_dir", "-o", default=None, type=str, required=True, help="The output directory.")
parser.add_argument("--emb_lgs", '-t', default='fr,es,de,zh,el,bg,ru,tr,ar,vi,th,hi,sw,ur', type=str, required=False, help="Embedded languages.")
parser.add_argument("--seed", '-sd', default=42, type=int, required=False, help="Random seed")
parser.add_argument("--sample_lgs", '-sl', default=2, type=int, required=False, help="Number of embedded languages per example.")
parser.add_argument("--sample_prob", '-sp', default=0.33, type=float, required=False, help="Probability of modifying a sentence.")
parser.add_argument("--perturb_prob", '-ptb', default=0.15, type=float, required=False, help="Probability of perturbing a word/phrase.")
parser.add_argument("--device", default='cuda', type=str, required=False, help="Device to use {'tpu', 'cuda', 'cpu'}.")
parser.add_argument("--lg_counts", default=None, type=str, required=False, help="Path to adversarial language distribution.")
parser.add_argument("--extract_phrases", '-ep', action='store_true', required=False, help="Extract phrases only.")
parser.add_argument("--phrase_alignments", '-pa', type=str, default=None, required=False, help="Path to extracted phrase alignments.")
parser.add_argument("--split", '-s', default='train', type=str, required=False, help="Use the train or test data as the source.")
parser.add_argument("--num_k", '-k', default=1, type=int, required=False, help="Number of perturbed examples per clean example.")
parser.add_argument("--gpu", '-g', default=0.25, type=float, required=False, help="GPU allocation per actor (% of one GPU). Total number of parallel actors is calculated using this value. Set to 0 to use CPU.")
args = parser.parse_args()


MTX_LG = 'en'

emb_lgs = args.emb_lgs.split(',')

# V100: 0.1428, A100:
USE_CUDA = torch.cuda.is_available() and args.gpu > 0
NUM_GPU_PER_ACTOR = args.gpu if USE_CUDA else 0
NUM_ACTOR_CPU_ONLY = 80
TOP_UP = 0 # Increase this value if number of generated examples is insufficient due to lack of candidates for some examples. We compensate by increasing k for other examples with more candidates.


@ray.remote(num_gpus=NUM_GPU_PER_ACTOR)
class CodemixActor(object):
    def __init__(self, mtx_lg, emb_lgs, lg_counts, device, actor_id, reference_translations=None):
        print(str(actor_id) + ' spawned')
        if args.phrase_alignments:
            self.mixer = CodeMixer(mtx_lg, emb_lgs, device, True)
        else:
            self.mixer = CodeMixer(mtx_lg, emb_lgs, device)

        self.refs = reference_translations
        self.lg_counts_dict = {k:v for k,v in lg_counts.items() if k in emb_lgs} if lg_counts else None
        self.lg_counts = list(self.lg_counts_dict.values()) if self.lg_counts_dict else None
        self.emb_lgs = list(self.lg_counts_dict.keys()) if self.lg_counts_dict else emb_lgs

    def mutate(self, batch, k, top_up=0):
        results = []
        for example in tqdm(batch):

            if example.get('preserved', False):
                results.append({'sentence1': example['sentence1'],
                               'sentence2': example['sentence2'],
                               'gold_label': example['gold_label'],
                               'preserved': 1})
                continue


            num_tries = 0
            results_set = set()

            adjusted_k = k+1 if top_up > 0 else k

            while len(results_set) < adjusted_k and num_tries < k*50:
                result = {}
                example['sentence1_phrases'] = {int(k): v for k,v in example['sentence1_phrases'].items()}
                example['sentence2_phrases'] = {int(k): v for k,v in example['sentence2_phrases'].items()}

                result['sentence1'] = self.mixer.generate_precomputed_alignments(example['sentence1'],
                                                                                 example['sentence1_phrases'],
                                                                                 args.perturb_prob)
                result['sentence1'] = result['sentence1'].replace(' ,', ',').replace(' .', '.')\
                                                         .replace(" '", "'").replace('( ', '(')\
                                                         .replace(' )', ')').replace(' ?', '?').replace(' !', '!')
                result['sentence2'] = self.mixer.generate_precomputed_alignments(example['sentence2'],
                                                                                 example['sentence2_phrases'],
                                                                                 args.perturb_prob)
                result['sentence2'] = result['sentence2'].replace(' ,', ',').replace(' .', '.')\
                                                         .replace(" '", "'").replace('( ', '(')\
                                                         .replace(' )', ')').replace(' ?', '?').replace(' !', '!')
                result['gold_label'] = example['gold_label']
                if result['sentence1'] != example['sentence1'] or result['sentence2'] != example['sentence2']:
                    result['preserved'] = 0
                else:
                    result['preserved'] = 1

                if (result['sentence1'], result['sentence2']) not in results_set:
                    results_set.add((result['sentence1'], result['sentence2']))
                    results.append(result)
                    if len(results_set) > k:
                        top_up -= 1
                num_tries += 1

        return results

    def extract_phrases(self, batch):
        results = []
        for i, example in enumerate(tqdm(batch)):
            text_a_phrases = {}
            text_b_phrases = {}
            random.seed(args.seed+i+len(example.text_a))
            if random.random() <= args.sample_prob:
                random.seed(args.seed+i+len(example.text_a))
                chosen_lgs = []
                tries = 0
                while len(list(filter(None, [self.refs[0][example.text_a][lg] for lg in chosen_lgs]))) < args.sample_lgs and tries < 5*args.sample_lgs:
                    tries += 1
                    chosen_lgs = random.choices(self.emb_lgs, weights=self.lg_counts, k=args.sample_lgs)
                text_a_phrases = self.mixer.get_phrases(example.text_a, self.refs[0][example.text_a], chosen_lgs)


            random.seed(args.seed+i+2*len(example.text_b))
            if random.random() <= args.sample_prob:
                random.seed(args.seed+i+2+len(example.text_b))
                chosen_lgs = [] #random.choices(self.emb_lgs, weights=self.lg_counts, k=args.sample_lgs)
                while len(list(filter(None, [self.refs[1][example.text_b][lg] for lg in chosen_lgs]))) < args.sample_lgs and tries < 5:
                    tries += 1
                    chosen_lgs = random.choices(self.emb_lgs, weights=self.lg_counts, k=args.sample_lgs)
                text_b_phrases = self.mixer.get_phrases(example.text_b, self.refs[1][example.text_b], chosen_lgs)
            if example.label == 'contradictory':
                example.label = 'contradiction'
            if text_a_phrases or text_b_phrases:
                preserved = 0
            else:
                preserved = 1
            results.append({'sentence1': example.text_a, 'sentence2': example.text_b,
                            'sentence1_phrases': text_a_phrases, 'sentence2_phrases': text_b_phrases,
                            'preserved': preserved, 'gold_label': example.label})

        return results


def _create_output_data(examples):
    #output = [['sentence1', 'sentence2', 'gold_label', 'preserved']]
    output = []
    for example in tqdm(examples, desc='Creating output'):
        output_line = [example['sentence1'], example['sentence2'], example['gold_label'], example['preserved']]
        output.append(output_line)
    return output


def _write_tsv(output, output_file):
    with open(output_file, "w", encoding="utf-8-sig") as f:
        writer = csv.writer(f, delimiter="\t", quotechar=None)
        for row in tqdm(output, desc='Writing output'):
            writer.writerow(row)

def get_examples(data_dir, split):
    if split == 'train':
        return processors['mnli']().get_train_examples(data_dir)
    elif split == 'test':
        return processors['mnli']().get_test_examples(data_dir)

def get_examples_w_phrases(data_file):
    examples = []
    with jsonlines.open(data_file, mode='r') as reader:
        for example in reader:
            examples.append(example)
    return examples

if args.phrase_alignments:
    reference_translations = None
    examples = get_examples_w_phrases(args.phrase_alignments)
else:
    reference_translations = [json.load(open('xnli-'+args.split+'-sentence1-reference-translations-en-head.json','r')),
                              json.load(open('xnli-'+args.split+'-sentence2-reference-translations-en-head','r'))]
    examples = get_examples(args.data, args.split)
    examples.reverse()

if args.lg_counts:
    weight_flag = '.weighted'
    lg_counts = json.load(open(args.lg_counts,'r'))
else:
    weight_flag = '.unweighted'
    lg_counts = None

output_path = Path(args.output_dir, 'codemixed_mnli.'+'_'.join(emb_lgs)+weight_flag)




args.device = 'cpu' if args.device == 'cuda' and not torch.cuda.is_available() else args.device

num_actors = int(torch.cuda.device_count() // NUM_GPU_PER_ACTOR) if args.device == 'cuda' else (64 if args.device == 'tpu' else NUM_ACTOR_CPU_ONLY)
print('Number of CodeMixers:', num_actors)

total_exs = len(examples)
print(total_exs)
len_per_batch = ceil(total_exs / num_actors)

batches = [examples[i:i+len_per_batch] for i in range(0, total_exs, len_per_batch)]

ray.init()



actors = ActorPool([CodemixActor.remote(MTX_LG, emb_lgs, lg_counts, args.device, i, reference_translations)
                   for i in range(num_actors)])
start = time.time()


if args.phrase_alignments:
    results = list(actors.map(lambda actor, batch: actor.mutate.remote(batch, args.num_k, TOP_UP), batches))
    time_taken = time.time() - start
    condition_name = '.'.join(args.phrase_alignments.split('/')[-1].split('.')[1:-1])
else:
    phrase_results = list(actors.map(lambda actor, batch: actor.extract_phrases.remote(batch), batches))

    output_path.mkdir(parents=True, exist_ok=True)


    time_taken = time.time() - start
    condition_name = 'seed-' + str(args.seed) + '.smp_lgs-' + str(args.sample_lgs) + '.smp_prob-'+ str(args.sample_prob)
    combined_phrase_results = [ex for batch in phrase_results for ex in batch]
    output_file_phrases = Path(output_path, args.split+'-extracted_phrases.' + condition_name + '.jsonl')
    with jsonlines.open(output_file_phrases, mode='w') as writer:
        for result in tqdm(combined_phrase_results, desc='Writing output'):
            writer.write(result)
    if not args.extract_phrases:
        results = list(actors.map(lambda actor, batch: actor.mutate.remote(batch, args.num_k), phrase_results))


if not args.extract_phrases:
    results = [ex for batch in results for ex in batch]
    output_path = Path(output_path, condition_name + '.ptb_prob-' + str(args.perturb_prob) + '.k-' + str(args.num_k))
    output_path.mkdir(parents=True, exist_ok=True)
    if args.split == 'train':
        output_file = Path(output_path, 'train.tsv')
    elif args.split == 'test':
        output_file = Path(output_path, 'test_matched.tsv')
    _write_tsv(_create_output_data(results), output_file)

print("Time taken:", time_taken / 60)
