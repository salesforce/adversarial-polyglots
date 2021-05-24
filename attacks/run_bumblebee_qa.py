import os, argparse, ray, torch, time, json, copy
from pathlib import Path
from bumblebee_align import BumblebeeQuestionAnsweringHF
from tqdm import tqdm
from ray.util import ActorPool
from math import ceil

parser = argparse.ArgumentParser()
parser.add_argument("--data", "-d", default=None, type=str, required=True, help="The input data file, e.g., 'data/xquad.en.json'.")
parser.add_argument("--model", "-m", type=str, required=True)
parser.add_argument("--output_dir", "-o", default=None, type=str, required=True, help="The output directory.")
parser.add_argument("--beam", '-b', default=0, type=int, required=False, help="Beam size.")
parser.add_argument("--tgt_langs", '-t', default='ar,de,el,es,hi,ru,th,tr,vi,zh', type=str, required=False, help="Embedded languages.")
parser.add_argument("--gpu", '-g', default=0.25, type=float, required=False, help="GPU allocation per actor (% of one GPU). Total number of parallel actors is calculated using this value. Set to 0 to use CPU.")
args = parser.parse_args()


USE_CUDA = torch.cuda.is_available() and args.gpu > 0
NUM_GPU_PER_ACTOR = args.gpu if USE_CUDA else 0
SRC_LANG = 'en'
TGT_LANGS = args.tgt_langs.split(',')

@ray.remote(num_gpus=NUM_GPU_PER_ACTOR)
class PolyglotActor(object):
    def __init__(self, model, src_lang, tgt_langs, reference_translations):
        self.polyglot = BumblebeeQuestionAnsweringHF(model, src_lang, tgt_langs, use_cuda=USE_CUDA)
        self.reference_translations = reference_translations

    def mutate(self, batch, beam):
        score = 0
        perturbed_questions = {"highest": {},
                               "lowest": {}}


        for question_dict, context in tqdm(batch):
            refs = self.reference_translations[question_dict['question']]

            question, text_label, f1, lowest_question, lowest_text_label, lowest_f1 = self.polyglot.generate(question_dict,
                                                                                               context,
                                                                                               reference_translations=refs,
                                                                                               beam_size=beam)
            perturbed_questions["highest"][question_dict['id']] = question
            perturbed_questions["lowest"][question_dict['id']] = lowest_question
            score += f1
        return perturbed_questions, score


def _create_output_data(questions, input_data_path):
    input_data = json.load(open(input_data_path))
    data = copy.deepcopy(input_data)
    for i, article in enumerate(input_data['data']):
        for j, paragraph in enumerate(article['paragraphs']):
            for k, qa in enumerate(paragraph['qas']):
                data['data'][i]['paragraphs'][j]['qas'][k]['question'] = questions[qa['id']]
    return data

def get_examples(data_path):
    input_data = json.load(open(data_path))
    examples = []
    for i, article in enumerate(input_data['data']):
        for j, paragraph in enumerate(article['paragraphs']):
            for k, qa in enumerate(paragraph['qas']):
                examples.append((qa, paragraph['context']))
    return examples


output_path = Path(args.output_dir,
                   'bumblebee-squad.' + \
                   args.model.strip('/').split('/')[-1] + \
                   '.' + '_'.join(TGT_LANGS) + \
                   '.beam-' + str(args.beam) + '.equiv_constr'+ '.pipe')


output_file = 'bumblebee.' + args.data.strip('/').split('/')[-1].split('.')[0]

output_file = str(Path(output_path, output_file))
print('Output file path:', output_file)


examples = get_examples(args.data)

reference_translations = json.load(open('xquad-question-reference-translations-en_head-th_zh_ws_tokenized.json','r'))


num_actors = int(torch.cuda.device_count() // NUM_GPU_PER_ACTOR) if USE_CUDA else 15
print('Number of Polyglots:', num_actors)

total_exs = len(examples)
print(total_exs)
len_per_batch = ceil(total_exs / num_actors)

batches = [examples[i:i+len_per_batch] for i in range(0, total_exs, len_per_batch)]

ray.init()
actors = ActorPool([PolyglotActor.remote(args.model, SRC_LANG, TGT_LANGS, reference_translations)
                   for i in range(num_actors)])


start = time.time()
results, scores = map(list, zip(*actors.map(lambda actor, batch: actor.mutate.remote(batch, args.beam), batches)))
time_taken = time.time() - start
results = {'highest': {qid: ex for batch in results for qid, ex in batch['highest'].items()},
           'lowest': {qid: ex for batch in results for qid, ex in batch['lowest'].items()}}

print("F1:", str(sum(scores)/total_exs * 100))
print("Time taken:", time_taken / 60)
print("Output: ", str(output_path))
output_path.mkdir(parents=True, exist_ok=True)

for loss in ['highest', 'lowest']:
    with open(output_file+'.'+loss+'.json', 'w') as outf:
        json.dump(_create_output_data(results[loss], args.data), outf, ensure_ascii=False, indent=4)

with open(str(Path(output_path, 'results.txt')), 'w') as t:
    t.write('F1: '+str(sum(scores)/total_exs * 100)+'\n')
    t.write("Time taken: "+str(time_taken / 60)+'\n')
