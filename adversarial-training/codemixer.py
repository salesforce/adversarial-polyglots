# -*- coding: utf-8 -*-
import random, json, string, warnings
from typing import Union, List, Set, Dict
from simalign import SentenceAligner
from nltk.translate.phrase_based import phrase_extraction


class CodeMixer(object):
    def __init__(self, matrix_lg: str, embedded_lgs: Union[List[str],Set[str]], device='cuda', precomputed_phrases=False):
        warnings.filterwarnings("ignore", category=FutureWarning)
        if not precomputed_phrases:
            self.aligner = SentenceAligner(model="xlmr", token_type="bpe", matching_methods="m", device=device)
        self.rtl_lgs = {'ar', 'he'}
        self.matrix_lg = matrix_lg
        self.embedded_lgs = set(embedded_lgs)


    def get_phrases(self, matrix_sentence: str, translations: Dict[str,str],
                    sample_lgs: Union[List[str],Set[str]]=None):
        if not sample_lgs:
            sample_lgs = self.embedded_lgs
        filtered_translations = {k: v for k,v in translations.items() if k in sample_lgs}
        matrix_tokens = matrix_sentence.split()
        phrases = []
        for lg, embedded_sentence in filtered_translations.items():
            tokenized_embedded = embedded_sentence.split()
            alignments = self.aligner.get_word_aligns(matrix_tokens, tokenized_embedded)
            candidate_phrases = phrase_extraction(matrix_sentence, embedded_sentence, alignments['mwmf'], 4)
            candidate_phrases = [(p[0], p[1], p[2], p[3], lg) for p in candidate_phrases if p[2][0] not in string.punctuation]
            if lg == 'zh':
                candidate_phrases = [(p[0], p[1], p[2], p[3].replace(' ', ''), p[4]) for p in candidate_phrases]
            if lg == 'th':
                candidate_phrases = [(p[0], p[1], p[2], p[3].replace('   ', ' ').replace(' ,', ','), p[4]) for p in candidate_phrases]

            phrases += candidate_phrases

        sorted_phrases = sorted(phrases, key=lambda x: (x[0][0], -x[0][1]))


        grouped_phrases = {i:[] for i in range(len(matrix_tokens))}
        for phrase in sorted_phrases:
            grouped_phrases[phrase[0][0]].append(phrase)

        return grouped_phrases

    def swap_phrase(self, tokens, replace_start_idx, replace_end_idx, to_replace):
        return tokens[0:replace_start_idx] + [to_replace] + tokens[replace_end_idx:]

    def get_weights(self, lg_counts):
        filtered_lg_counts = {k: v for k,v in lg_counts.items() if k in self.embedded_lgs or k == self.matrix_lg}

        total_count = sum(filtered_lg_counts.values())
        return {k: v/total_count for k,v in filtered_lg_counts.items()}

    def generate(self, sentence, reference_translations, probability=0.15, lg_counts: Dict[str,int]=None):
        phrases = self.get_phrases(sentence, reference_translations)
        return generate_precomputed_alignments(sentence, phrases, probability, lg_counts)

    def generate_precomputed_alignments(self, sentence, phrase_alignments, probability=0.15):#, lg_counts: Dict[str,int]=None):
        tokens = sentence.split()

        token_length = len(tokens)
        pos = 0
        prev_lg = self.matrix_lg
        prev_replacement_pos = pos
        while pos < token_length:
            candidates = phrase_alignments.get(pos)

            pos += 1
            if random.random() >= probability or not candidates:
                prev_lg = self.matrix_lg
                continue

            eligible_candidates = []

            for candidate in candidates:
                phrase_to_replace = candidate[2]
                replacement = candidate[3]
                replacement_lg = candidate[4]
                replace_start_idx = candidate[0][0] - token_length
                replace_end_idx = candidate[0][1] - token_length


                if phrase_to_replace.split() != tokens[replace_start_idx:replace_end_idx]:
                    continue
                if replacement_lg not in self.rtl_lgs and replacement_lg == prev_lg and candidate[1][1] <= prev_replacement_pos:
                    continue
                eligible_candidates.append(candidate)

            if eligible_candidates:
                chosen_candidate = random.choice(eligible_candidates)

                replacement_lg = chosen_candidate[4]
                replacement = chosen_candidate[3]
                replace_start_idx = chosen_candidate[0][0] - token_length
                replace_end_idx = chosen_candidate[0][1] - token_length

                tokens = self.swap_phrase(tokens, replace_start_idx, replace_end_idx, replacement)
                prev_lg = replacement_lg
                prev_replacement_pos = pos
                pos = max(replace_end_idx, pos)

            else:
                prev_lg = self.matrix_lg
                continue

        return ' '.join(tokens)
