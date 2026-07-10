#!/usr/bin/env python3
'''kskjdkd'''
from nltk.translate.bleu_score import sentence_bleu


def uni_bleu(references, sentence):
    '''def'''
    score = sentence_bleu(references, sentence, weights=(1, 0, 0, 0))
    return score
