from abc import ABCMeta, abstractmethod
import json, random, torch
from typing import Union, List, Set, Dict
from tokenizers.pre_tokenizers import BertPreTokenizer
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from transformers import BartTokenizer, BartTokenizerFast, RobertaTokenizer, RobertaTokenizerFast, XLMRobertaTokenizer
from torch.nn import CrossEntropyLoss
from sortedcontainers import SortedList
from opencc import OpenCC


class PolyTokenizer:
    contractions = {"won't": "will not",
                    "n't": " not",
                    "'ll": " will",
                    "I'm": "I am",
                    "she's": "she is",
                    "he's": "he is",
                    "'re": " are"
                   }

    def __init__(self):
        self.tokenizer = BertPreTokenizer()

    def expand_en_contractions(self, sentence):
        new_sentence = sentence
        for contraction, expansion in self.contractions.items():
            new_sentence = new_sentence.replace(contraction, expansion)
        return new_sentence

    def tokenize(self, sentence):
        expanded = self.expand_en_contractions(sentence)
        tokenized = self.tokenizer.pre_tokenize(expanded)
        tokens, char_ids = list(zip(*tokenized))

        return list(tokens), char_ids


    def detokenize(self, tokens, char_ids):
        out_tokens = []
        for i, token in enumerate(tokens):
            if i != 0 and char_ids[i-1][1] != char_ids[i][0]:
                out_tokens.append(' ')
            out_tokens.append(token)

        return ''.join(out_tokens)


DEFAULT_IGNORE_WORDS = {'is','was','am','are','were','the','a', 'that'}


class PolyglossBase(metaclass=ABCMeta):
    def __init__(self,
                 source_lg: str,
                 target_lgs: Union[List[str],Set[str]],
                 map_path: str,
                 ignore_words: Union[List[str],Set[str]]=DEFAULT_IGNORE_WORDS
                ):
        with open(map_path, 'r') as f:
            self.word_map = json.load(f)
        self.source_lg = source_lg
        self.target_lgs = set(target_lgs)
        self.ignore_words = set(ignore_words)
        self.polytokenizer = PolyTokenizer()
        self.cc = OpenCC('s2t')

    @abstractmethod
    def generate(self):
         pass

    def get_candidates(self, word, reference_translations=None):
        all_lgs_cands = self.word_map.get(word)
        if not all_lgs_cands:
            return [word]
        candidates = all_lgs_cands.items()

        filtered_candidates = [word]
        for lg, lst in candidates:
            if lg not in self.target_lgs:
                continue
            for w in lst:
                if not reference_translations or w in reference_translations[lg]:
                    filtered_candidates.append(w)
                    if lg =='zh':
                        trad = self.cc.convert(w)
                        if trad != w: filtered_candidates.append(trad)
        return filtered_candidates


class PolyglossPairSequenceClassification(PolyglossBase):
    def __init__(self, source_lg, target_lgs, map_path, labels: List[str]):
        super().__init__(source_lg, target_lgs, map_path)
        self.labels = labels

    def is_flipped(self, predicted, label):
        return predicted != label

    def generate(self, sentence1, sentence2, label_text, beam_size=1,
                 reference_translations: Union[List[Dict[str,Dict[str,str]]],Dict[str,Dict[str,str]]]=None):
        assert label_text in self.labels
        label = self.labels.index(label_text)
        orig_s1_tokens, orig_s1_char_ids = self.polytokenizer.tokenize(sentence1)
        orig_s2_tokens, orig_s2_char_ids = self.polytokenizer.tokenize(sentence2)
        num_queries = 1
        original_loss, init_predicted = self.get_loss(sentence1, sentence2, label)

        if self.is_flipped(init_predicted, label):
            return sentence1, sentence2, self.labels[init_predicted], num_queries, sentence1, sentence2, self.labels[init_predicted]

        # search

        # init beam
        beam = SortedList()
        successful_candidates = SortedList()

        s1_token_length = len(orig_s1_tokens)
        s2_token_length = len(orig_s2_tokens)


        if reference_translations:
            s1_reference_translations = reference_translations[0]
            s2_reference_translations = reference_translations[1]
        else:
            s1_reference_translations = None
            s2_reference_translations = None

        # (loss, pos1, pos2, predicted_label, s1_tokens, s2_tokens)
        beam.add((original_loss, 0, 0, init_predicted, orig_s1_tokens, orig_s2_tokens))

        while beam:
            prev_loss, curr_pos1, curr_pos2, prev_predicted, prev_s1_tokens, prev_s2_tokens = beam.pop()
            s2_candidates = None

            s1_token_to_modify = prev_s1_tokens[curr_pos1]
            if s1_token_to_modify not in self.ignore_words:
                s1_candidates = self.get_candidates(s1_token_to_modify, s1_reference_translations)
            else:
                s1_candidates = [s1_token_to_modify]


            s2_token_to_modify = prev_s2_tokens[curr_pos2]
            if s2_token_to_modify not in self.ignore_words:
                s2_candidates = self.get_candidates(s2_token_to_modify, s2_reference_translations)
            else:
                s2_candidates = [s2_token_to_modify]


            for s1_candidate in s1_candidates:
                new_s1_tokens = prev_s1_tokens.copy()
                if new_s1_tokens[curr_pos1] == s1_candidate:
                    if curr_pos1+1 < s1_token_length:
                        beam.add((prev_loss, curr_pos1+1, curr_pos2, prev_predicted, prev_s1_tokens, prev_s2_tokens))
                    continue
                new_s1_tokens[curr_pos1] = s1_candidate
                s1_perturbed = self.polytokenizer.detokenize(new_s1_tokens, orig_s1_char_ids)
                s2_prev = self.polytokenizer.detokenize(prev_s2_tokens, orig_s2_char_ids)
                new_loss, new_predicted = self.get_loss(s1_perturbed, s2_prev, label)
                if self.is_flipped(new_predicted, label):
                    successful_candidates.add((new_loss, new_predicted, new_s1_tokens, prev_s2_tokens))
                if curr_pos1+1 < s1_token_length:
                    beam.add((new_loss, curr_pos1+1, curr_pos2, new_predicted, new_s1_tokens, prev_s2_tokens))
                num_queries += 1

            for s2_candidate in s2_candidates:
                new_s2_tokens = prev_s2_tokens.copy()
                if new_s2_tokens[curr_pos2] == s2_candidate:
                    if curr_pos2+1 < s2_token_length:
                        beam.add((prev_loss, curr_pos1, curr_pos2+1, prev_predicted, prev_s1_tokens, prev_s2_tokens))
                    continue
                new_s2_tokens[curr_pos2] = s2_candidate
                s1_prev = self.polytokenizer.detokenize(prev_s1_tokens, orig_s1_char_ids)
                s2_perturbed = self.polytokenizer.detokenize(new_s2_tokens, orig_s2_char_ids)
                new_loss, new_predicted = self.get_loss(s1_prev, s2_perturbed, label)
                if self.is_flipped(new_predicted, label):
                    successful_candidates.add((new_loss, new_predicted, prev_s1_tokens, new_s2_tokens))
                num_queries += 1
                if curr_pos2+1 < s2_token_length:
                    beam.add((new_loss, curr_pos1, curr_pos2+1, new_predicted, prev_s1_tokens, new_s2_tokens))

            beam = SortedList(beam[-beam_size:]) # trim beam

        if successful_candidates:
            _, final_predicted, final_s1_tokens, final_s2_tokens = successful_candidates[-1]
            sentence1 = self.polytokenizer.detokenize(final_s1_tokens, orig_s1_char_ids)
            sentence2 = self.polytokenizer.detokenize(final_s2_tokens, orig_s2_char_ids)
            _, lowest_final_predicted, lowest_final_s1_tokens, lowest_final_s2_tokens = successful_candidates[0]
            lowest_sentence1 = self.polytokenizer.detokenize(lowest_final_s1_tokens, orig_s1_char_ids)
            lowest_sentence2 = self.polytokenizer.detokenize(lowest_final_s2_tokens, orig_s2_char_ids)
        else:
            final_predicted = init_predicted
            lowest_sentence1 = sentence1
            lowest_sentence2 = sentence2
            lowest_final_predicted = init_predicted

        return sentence1, sentence2, self.labels[final_predicted], num_queries, lowest_sentence1, lowest_sentence2, self.labels[lowest_final_predicted]


class PolyglossPairSequenceClassificationHF(PolyglossPairSequenceClassification):
    def __init__(self, model_path, source_lg: str,
                 target_lgs: Union[List[str],Set[str]],
                 map_path: str,
                 labels: List[str],
                 is_nli=False,
                 use_cuda=True):
        super().__init__(source_lg, target_lgs, map_path, labels)
        if torch.cuda.is_available() and use_cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        config = AutoConfig.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if is_nli and self.tokenizer.__class__ in (RobertaTokenizer,
                                                    RobertaTokenizerFast,
                                                    XLMRobertaTokenizer,
                                                    BartTokenizer,
                                                    BartTokenizerFast):
            # hack to handle roberta models from huggingface
            labels[1], labels[2] = labels[2], labels[1]

        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

        self.model.eval()
        self.model.to(self.device)
        self.loss_fn = CrossEntropyLoss()

    def get_loss(self, sentence1, sentence2, label, max_seq_len=128):
        logits, _ = self.model_predict(sentence1, sentence2, max_seq_len)
        label_tensor = torch.tensor([label]).to(self.device)
        loss = self.loss_fn(logits, label_tensor)

        return loss.item(), logits.argmax().item()

    def model_predict(self, sentence1, sentence2, max_seq_len=512):
        inputs = self.tokenizer.encode_plus(sentence1, sentence2, add_special_tokens=True, max_length=max_seq_len, truncation=True)
        input_ids = torch.tensor(inputs["input_ids"]).unsqueeze(0).to(self.device)
        if "token_type_ids" in inputs.keys():
            token_type_ids = torch.tensor(inputs["token_type_ids"]).unsqueeze(0).to(self.device)
        else:
            # handle XLM-R
            token_type_ids = torch.tensor([0]*len(input_ids)).unsqueeze(0).to(self.device)

        outputs = self.model(input_ids,token_type_ids=token_type_ids)
        logits = outputs[0]
        return logits, self.labels[logits.argmax().item()]
