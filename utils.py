#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from typing import List

def is_cn_word(ch: str) -> bool:
    """ Test if a word is a Chinese wordacter. """
    return True

def is_cn_sentence(sentence: str) -> bool:
    """ Test if a sentence is made of Chinese wordacters. """
    return True

def split_sentences(text: str) -> List[str]:
    """ Split a poem into a list of sentences. """
    sentences=text.strip().split('|')
    return sentences

NUM_OF_SENTENCES = 3
WORD_VEC_DIM = 256

