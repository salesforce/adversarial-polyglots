# -*- coding: utf-8 -*-
from abc import ABCMeta, abstractmethod
import json, random, torch
from typing import Union, List, Set, Dict
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForQuestionAnswering, pipeline
from transformers import BartTokenizer, BartTokenizerFast, RobertaTokenizer, RobertaTokenizerFast, XLMRobertaTokenizer
from torch.nn import CrossEntropyLoss
from sortedcontainers import SortedKeyList
from math import copysign
from simalign import SentenceAligner
from nltk.translate.phrase_based import phrase_extraction
import string, warnings
from squad_utils import f1_max

DEFAULT_IGNORE_WORDS = {'is','was','am','are','were','the','a', 'that'}

class BumblebeeBase(metaclass=ABCMeta):
    def __init__(self,
                 source_lg: str,
                 target_lgs: Union[List[str],Set[str]],
                 ignore_words: Union[List[str],Set[str]]=DEFAULT_IGNORE_WORDS,
                 transliteration_map: Dict[str,str]=None
                ):

        self.aligner = SentenceAligner(model="xlmr", token_type="bpe", matching_methods="m")

        self.rtl_lgs = {'ar', 'he'}
        self.source_lg = source_lg
        self.target_lgs = set(target_lgs)
        self.ignore_words = set(ignore_words)
        self.transliteration_map = transliteration_map


    @abstractmethod
    def generate(self):
         return

    def transliterate_if_hindi(self, phrase, lg):
        if self.transliteration_map and lg == 'hi':
            return self.transliterate(phrase)
        else:
            return phrase

    def transliterate(self, phrase):
        words = phrase.split()
        new_words = []
        for word in words:
            if word in self.transliteration_map:
                new_words.append(self.transliteration_map[word])
            else:
                new_words.append(word)
        return ' '.join(new_words)


    def swap_phrase_simple(self, sentence, to_be_replaced, to_replace):
        return (' ' + sentence + ' ').replace(' ' + to_be_replaced + ' ', ' ' + to_replace + ' ', 1).strip()

    def swap_phrase(self, tokens, replace_start_idx, replace_end_idx, to_replace):
        return ' '.join(tokens[0:replace_start_idx] + [to_replace] + tokens[replace_end_idx:])


    def swap_phrase_w_split_bwd(self, sentence, split_pos, to_be_replaced, to_replace):
        tokenized = sentence.split()
        back = ' '.join(tokenized[split_pos:])
        front_tok = tokenized[:split_pos]

        new_back_tok = (' ' + back + ' ').replace(' ' + to_be_replaced + ' ', ' ' + to_replace + ' ', 1).strip().split()

        return ' '.join(front_tok + new_back_tok)


    def filter_equiv_constraint(self, phrases):
        sorted_phrases = sorted(phrases, key=lambda x: (x[0][0], x[0][1]))
        forward_constrained_candidates = []
        curr_matrix_pos = sorted_phrases[0][0][1]
        curr_embedded_pos = sorted_phrases[0][1][1]
        for phrase in sorted_phrases:
            if phrase[0][0] > curr_matrix_pos and phrase[1][0] < curr_embedded_pos:
                continue
            curr_matrix_pos = phrase[0][1]
            curr_embedded_pos = phrase[1][1]
            forward_constrained_candidates.append(phrase)

        return forward_constrained_candidates

    def get_phrases(self, matrix_sentence: str, translations: Dict[str,str]):
        filtered_translations = {k: v for k,v in translations.items() if k in self.target_lgs}
        matrix_tokens = matrix_sentence.split()
        phrases = []
        for lg, embedded_sentence in filtered_translations.items():

            tokenized_embedded = embedded_sentence.split()
            alignments = self.aligner.get_word_aligns(matrix_tokens, tokenized_embedded)

            candidate_phrases = phrase_extraction(matrix_sentence, embedded_sentence, alignments['mwmf'], 4)
            # [matrix range, matrix text, embedded range, embedded text, embedded language]
            candidate_phrases = [(p[0], p[1], p[2], p[3], lg) for p in candidate_phrases if p[2][0] not in string.punctuation]
            if lg == 'zh':
                candidate_phrases = [(p[0], p[1], p[2], p[3].replace(' ', ''), p[4]) for p in candidate_phrases]
            if lg == 'th':
                candidate_phrases = [(p[0], p[1], p[2], p[3].replace('   ', ' ').replace(' ,', ','), p[4]) for p in candidate_phrases]
            if self.transliteration_map and lg == 'hi':
                candidate_phrases = [(p[0], p[1], p[2], self.transliterate(p[3]), p[4]) for p in candidate_phrases]

            phrases += candidate_phrases

        sorted_phrases = sorted(phrases, key=lambda x: (x[0][0], -x[0][1]))


        grouped_phrases = {i:[] for i in range(len(matrix_tokens))}
        for phrase in sorted_phrases:
            grouped_phrases[phrase[0][0]].append(phrase)

        return grouped_phrases


class BumblebeePairSequenceClassification(BumblebeeBase):
    def __init__(self, source_lg, target_lgs, labels: List[str], transliteration_map: Dict[str,str]=None):
        super().__init__(source_lg, target_lgs, transliteration_map=transliteration_map)
        self.labels = labels

    def is_flipped(self, predicted, label):
        return predicted != label

    def generate(self, sentence1, sentence2, label_text,
                 reference_translations: Union[List[Dict[str,Dict[str,str]]],Dict[str,Dict[str,str]]],
                 beam_size=1, early_terminate=False):
        assert label_text in self.labels
        warnings.filterwarnings("ignore", category=FutureWarning)
        label = self.labels.index(label_text)
        orig_s1_tokens = sentence1.split()
        orig_s2_tokens = sentence2.split()
        num_queries = 1
        original_loss, init_predicted = self.get_loss(sentence1, sentence2, label)

        if self.is_flipped(init_predicted, label):
            return sentence1, sentence2, self.labels[init_predicted], {self.source_lg: -1}, num_queries, sentence1, sentence2, self.labels[init_predicted], {self.source_lg: -1}

        # search

        s1_reference_translations = reference_translations[0]
        s2_reference_translations = reference_translations[1]

        s1_phrases = self.get_phrases(sentence1, s1_reference_translations)
        s2_phrases = self.get_phrases(sentence2, s2_reference_translations)


        # init beam

        successful_candidates = SortedKeyList(key=lambda x: x[:-1])

        s1_token_length = len(orig_s1_tokens)
        s2_token_length = len(orig_s2_tokens)


        # (loss, pos1, pos2, predicted_label, s1_tokens, s2_tokens,
        #  lg1, prev_s1_replacement_token_positions, lg2, prev_s2_replacement_token_positions)
        '''
        Start from the front
        '''
        init_lg_counts = {lg: 0 for lg in self.target_lgs}
        curr_beam = [(original_loss, 0, 0,
                      init_predicted, sentence1, sentence2,
                      self.source_lg, (0, 1),
                      self.source_lg, (0, 1),
                      init_lg_counts)]
        new_beam = SortedKeyList(key=lambda x: x[:-1])
        early_terminate_flag = False
        while curr_beam and (not early_terminate or not early_terminate_flag):
            for prev_loss, curr_pos1, curr_pos2, prev_predicted, prev_s1, prev_s2, prev_lg1, \
                prev_s1_replacement_pos, prev_lg2, prev_s2_replacement_pos, lg_counts in curr_beam:

                prev_s1_tokens = prev_s1.split()
                prev_s2_tokens = prev_s2.split()

                s1_candidates = s1_phrases.get(curr_pos1)
                s2_candidates = s2_phrases.get(curr_pos2)

                if curr_pos1+1 < s1_token_length:
                    new_beam.add((original_loss, curr_pos1+1, curr_pos2,
                                  init_predicted, prev_s1, prev_s2,
                                  self.source_lg, (curr_pos1+1, curr_pos1+2),
                                  prev_lg2, prev_s2_replacement_pos,
                                  lg_counts))
                if curr_pos2+1 < s2_token_length:
                    new_beam.add((original_loss, curr_pos1, curr_pos2+1,
                                  init_predicted, prev_s1, prev_s2,
                                  prev_lg1, prev_s1_replacement_pos,
                                  self.source_lg, (curr_pos2+1, curr_pos2+2),
                                  lg_counts))


                for s1_candidate in s1_candidates:
                    phrase_to_replace = s1_candidate[2]
                    replacement = s1_candidate[3]
                    replacement_lg = s1_candidate[4]
                    replace_start_idx = s1_candidate[0][0] - s1_token_length
                    replace_end_idx = s1_candidate[0][1] - s1_token_length

                    if phrase_to_replace.split() != prev_s1_tokens[replace_start_idx:replace_end_idx]:
                        continue

                    if replacement_lg not in self.rtl_lgs and replacement_lg == prev_lg1 and s1_candidate[1][1] <= prev_s1_replacement_pos[0]:
                        continue

                    s1_perturbed = self.swap_phrase(prev_s1_tokens, replace_start_idx, replace_end_idx, replacement)

                    new_loss, new_predicted = self.get_loss(s1_perturbed, prev_s2, label)
                    new_lg_counts = lg_counts.copy()
                    new_lg_counts[replacement_lg] += 1
                    if self.is_flipped(new_predicted, label):
                        successful_candidates.add((new_loss, new_predicted, s1_perturbed, prev_s2, new_lg_counts))
                        early_terminate_flag = True
                        if early_terminate: break
                    if curr_pos1+1 < s1_token_length:
                        new_beam.add((new_loss, curr_pos1+1, curr_pos2, new_predicted, s1_perturbed, prev_s2,
                                      replacement_lg, s1_candidate[1], prev_lg2, prev_s2_replacement_pos,
                                      new_lg_counts))
                    num_queries += 1

                for s2_candidate in s2_candidates:
                    phrase_to_replace = s2_candidate[2]
                    replacement = s2_candidate[3]
                    replacement_lg = s2_candidate[4]
                    replace_start_idx = s2_candidate[0][0] - s2_token_length
                    replace_end_idx = s2_candidate[0][1] - s2_token_length

                    if phrase_to_replace.split() != prev_s2_tokens[replace_start_idx:replace_end_idx]:
                        continue

                    if replacement_lg not in self.rtl_lgs and replacement_lg == prev_lg2 and s2_candidate[1][1] <= prev_s2_replacement_pos[0]:
                        continue

                    s2_perturbed = self.swap_phrase(prev_s2_tokens, replace_start_idx, replace_end_idx, replacement)

                    new_loss, new_predicted = self.get_loss(prev_s1, s2_perturbed, label)
                    new_lg_counts = lg_counts.copy()
                    new_lg_counts[replacement_lg] += 1
                    if self.is_flipped(new_predicted, label):
                        successful_candidates.add((new_loss, new_predicted, prev_s1, s2_perturbed, new_lg_counts))
                        early_terminate_flag = True
                        if early_terminate: break
                    if curr_pos2+1 > s2_token_length:
                        new_beam.add((new_loss, curr_pos1, curr_pos2+1, new_predicted, prev_s1, s2_perturbed,
                                      prev_lg1, prev_s1_replacement_pos, replacement_lg, s2_candidate[1],
                                      new_lg_counts))
                    num_queries += 1

                curr_beam = new_beam[-beam_size:] # trim beam
                new_beam = SortedKeyList(key=lambda x: x[:-1])

        if successful_candidates:
            _, final_predicted, sentence1, sentence2, final_lg_counts = successful_candidates[-1]
            _, lowest_final_predicted, lowest_sentence1, lowest_sentence2, lowest_final_lg_counts = successful_candidates[0]
        else:
            final_predicted = init_predicted
            lowest_sentence1 = sentence1
            lowest_sentence2 = sentence2
            lowest_final_predicted = init_predicted
            final_lg_counts = {self.source_lg: -1}
            lowest_final_lg_counts = {self.source_lg: -1}

        return sentence1, sentence2, self.labels[final_predicted], final_lg_counts, num_queries, lowest_sentence1, lowest_sentence2, self.labels[lowest_final_predicted], lowest_final_lg_counts


class BumblebeePairSequenceClassificationHF(BumblebeePairSequenceClassification):
    def __init__(self, model_path, source_lg: str,
                 target_lgs: Union[List[str],Set[str]],
                 labels: List[str],
                 is_nli=False,
                 use_cuda=True,
                 transliteration_map: Dict[str,str]=None):
        super().__init__(source_lg, target_lgs, labels, transliteration_map=transliteration_map)
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
            self.labels[1], self.labels[2] = self.labels[2], self.labels[1]

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


class BumblebeeQuestionAnswering(BumblebeeBase):
    def is_flipped(self, predicted, answer_texts: List[str]):
        return f1_max(predicted, answer_texts) == 0

    def generate(self, question_dict, context,
                 reference_translations: Dict[str,Dict[str,str]],
                 beam_size=1):
        warnings.filterwarnings("ignore", category=FutureWarning)
        question = question_dict['question']
        answer_dict = {}
        gold_starts = [ans['answer_start'] for ans in question_dict['answers']]
        gold_texts = [ans['text'] for ans in question_dict['answers']]
        gold_ends = [gold_starts[i]+len(text) for i, text in enumerate(gold_texts)]
        answer_dict['gold_char_spans'] = list(zip(gold_starts, gold_ends))

        answer_dict['gold_texts'] = gold_texts


        num_queries = 1
        original_loss, init_predicted = self.get_loss(question, context, answer_dict)

        # search
        question_phrases = self.get_phrases(question, reference_translations)

        # init beam
        beam = SortedList()
        successful_candidates = SortedList()
        partially_successful_candidates = SortedList()

        orig_qn_tokens = question.split()
        qn_token_length = len(orig_qn_tokens)

        # (loss, pos, predicted_answer, qn_tokens)
        '''
        Start from the front
        '''
        beam.add((original_loss, 0, init_predicted, question, self.source_lg, (0, 1)))

        while beam:
            prev_loss, curr_pos, prev_predicted, prev_qn, prev_lg, prev_replacement_pos = beam.pop()
            prev_qn_tokens = prev_qn.split()

            qn_candidates = question_phrases.get(curr_pos)

            if curr_pos+1 < qn_token_length:
                beam.add((original_loss, curr_pos+1, init_predicted, prev_qn, self.source_lg, (curr_pos+1, curr_pos+2)))

            for qn_candidate in qn_candidates:
                phrase_to_replace = qn_candidate[2]
                replacement = qn_candidate[3]
                replacement_lg = qn_candidate[4]
                replace_start_idx = qn_candidate[0][0] - qn_token_length
                replace_end_idx = qn_candidate[0][1] - qn_token_length

                if phrase_to_replace.split() != prev_qn_tokens[replace_start_idx:replace_end_idx]:
                    continue

                if replacement_lg not in self.rtl_lgs and replacement_lg == prev_lg and qn_candidate[1][1] <= prev_replacement_pos[0]:
                    continue


                qn_perturbed = self.swap_phrase(prev_qn_tokens, replace_start_idx, replace_end_idx, replacement)

                new_loss, new_predicted = self.get_loss(qn_perturbed, context, answer_dict)
                if self.is_flipped(new_predicted, answer_dict['gold_texts']):
                    successful_candidates.add((new_loss, new_predicted, qn_perturbed))
                elif f1_max(new_predicted, answer_dict['gold_texts']) < 1.0:
                    partially_successful_candidates.add((new_loss, new_predicted, qn_perturbed))
                if curr_pos+1 < qn_token_length:
                    beam.add((new_loss, curr_pos+1, new_predicted, qn_perturbed, replacement_lg, qn_candidate[1]))
                num_queries += 1

            beam = SortedList(beam[-beam_size:]) # trim beam

        if successful_candidates:
            _, final_predicted, question = successful_candidates[-1]
            _, lowest_final_predicted, lowest_question = successful_candidates[0]
        elif partially_successful_candidates:
            _, final_predicted, question = partially_successful_candidates[-1]
            _, lowest_final_predicted, lowest_question = partially_successful_candidates[0]
        else:
            final_predicted = init_predicted
            lowest_question = question
            lowest_final_predicted = init_predicted

        return question, final_predicted, f1_max(final_predicted, answer_dict['gold_texts']), lowest_question, lowest_final_predicted, f1_max(lowest_final_predicted, answer_dict['gold_texts'])


    def bwd_generate(self, question_dict, context,
                 reference_translations: Dict[str,Dict[str,str]],
                 beam_size=1):

        question = question_dict['question']
        answer_dict = {}
        gold_starts = [ans['answer_start'] for ans in question_dict['answers']]
        gold_texts = [ans['text'] for ans in question_dict['answers']]
        gold_ends = [gold_starts[i]+len(text) for i, text in enumerate(gold_texts)]
        answer_dict['gold_char_spans'] = list(zip(gold_starts, gold_ends))

        answer_dict['gold_texts'] = gold_texts


        num_queries = 1
        original_loss, init_predicted = self.get_loss(question, context, answer_dict)


        # search
        question_phrases = self.get_phrases(question, reference_translations)

        # init beam
        beam = SortedList()
        successful_candidates = SortedList()
        partially_successful_candidates = SortedList()

        orig_qn_tokens = question.split()
        qn_token_length = len(orig_qn_tokens)

        # (loss, pos, predicted_answer, qn_tokens)
        '''
        Start from the back
        '''
        beam.add((original_loss, qn_token_length-1, init_predicted, question))

        while beam:
            prev_loss, curr_pos, prev_predicted, prev_qn = beam.pop()
            prev_qn_tokens = prev_qn.split()

            qn_candidates = question_phrases.get(curr_pos)

            if curr_pos-1 > -1:
                beam.add((original_loss, curr_pos-1, init_predicted, prev_qn))


            for qn_candidate in qn_candidates:
                phrase_to_replace = qn_candidate[2]
                replacement = qn_candidate[3]

                if phrase_to_replace not in prev_qn:
                    continue

                qn_perturbed = self.swap_phrase_w_split(prev_qn, curr_pos, phrase_to_replace, replacement)

                new_loss, new_predicted = self.get_loss(qn_perturbed, context, answer_dict)
                if self.is_flipped(new_predicted, answer_dict['gold_texts']):
                    successful_candidates.add((new_loss, new_predicted, qn_perturbed))
                elif f1_max(new_predicted, answer_dict['gold_texts']) < 0.5:
                    partially_successful_candidates.add((new_loss, new_predicted, qn_perturbed))
                if curr_pos-1 > -1:
                    beam.add((new_loss, curr_pos-1, new_predicted, qn_perturbed))
                num_queries += 1

            beam = SortedList(beam[-beam_size:]) # trim beam

        if successful_candidates:
            _, final_predicted, question = successful_candidates[-1]
            _, lowest_final_predicted, lowest_question = successful_candidates[0]
        elif partially_successful_candidates:
            _, final_predicted, question = partially_successful_candidates[-1]
            _, lowest_final_predicted, lowest_question = partially_successful_candidates[0]
        else:
            final_predicted = init_predicted
            lowest_question = question
            lowest_final_predicted = init_predicted

        return question, final_predicted, f1_max(final_predicted, answer_dict['gold_texts']), lowest_question, lowest_final_predicted, f1_max(lowest_final_predicted, answer_dict['gold_texts'])


    def get_lowest_loss(self, start_logits_tensor, end_logits_tensor, gold_spans):
        start_logits_tensor = start_logits_tensor.to('cpu')
        end_logits_tensor = end_logits_tensor.to('cpu')
        target_tensors = [(torch.tensor([gold_start]), torch.tensor([gold_end])) \
                              for gold_start, gold_end in gold_spans]

        losses = []
        for target_start, target_end in target_tensors:
            avg_loss = (self.loss_fn(start_logits_tensor, target_start) \
                        + self.loss_fn(end_logits_tensor, target_end))/2
            losses.append(avg_loss)
        return min(losses).item()


class BumblebeeQuestionAnsweringHF(BumblebeeQuestionAnswering):
    def __init__(self, model_path, source_lg: str,
                 target_lgs: Union[List[str],Set[str]],
                 use_cuda=True):
        super().__init__(source_lg, target_lgs)
        if torch.cuda.is_available() and use_cuda:
            device = 'cuda'
        else:
            device = 'cpu'
        self.qa_pipeline = pipeline('question-answering', model=model_path,
                                    tokenizer=model_path, device=device)


    def get_loss(self, question, context, answer_dict):
        result = self.qa_pipeline(question=question, context=context)
        return copysign(result['score'], -f1_max(result['answer'], answer_dict['gold_texts'])), result['answer']
