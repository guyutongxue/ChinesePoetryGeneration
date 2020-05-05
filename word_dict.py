#! /usr/bin/env python3
#-*- coding:utf-8 -*-

from paths import word_dict_path,check_uptodate
import pickle as pkl
from singleton import Singleton

MAX_DICT_SIZE = 100000
word_dict_storefile=r'./raw/raw_dict.pkl'


def start_of_sentence():
    return '^'

def end_of_sentence():
    return '$'

def _gen_word_dict():
    print("Generating dictionary from corpus ...")
    
    # import word frequencies.
    words=pkl.load(open(word_dict_storefile,'rb'))
    # Sort in decreasing order of frequency.
    cnt2word = sorted(words.items(), key = lambda x: -x[1])

    # Store most popular words into the file.
    with open(word_dict_path, 'w',encoding='utf-8') as fout:
        for i in range(min(MAX_DICT_SIZE - 2, len(cnt2word))):
            fout.write(cnt2word[i][0]+'|')

class wordDict(Singleton):

    def __init__(self):
        if not check_uptodate(word_dict_path):
            _gen_word_dict()
        self._int2word = []
        self._word2int = dict()
        # Add start-of-sentence symbol.
        self._int2word.append(start_of_sentence())
        self._word2int[start_of_sentence()] = 0
        with open(word_dict_path, 'r',encoding='utf-8') as fin:
            idx = 1
            for ch in fin.read().strip().split('|'):
                self._int2word.append(ch)
                self._word2int[ch] = idx
                idx += 1
        # Add end-of-sentence symbol.
        self._int2word.append(end_of_sentence())
        self._word2int[end_of_sentence()] = len(self._int2word) - 1

    def word2int(self, ch):
        if ch not in self._word2int:
            return -1
        return self._word2int[ch]

    def int2word(self, idx):
        return self._int2word[idx]

    def __len__(self):
        return len(self._int2word)

    def __iter__(self):
        return iter(self._int2word)

    def __contains__(self, ch):
        return ch in self._word2int

# For testing purpose.
if __name__ == '__main__':
    word_dict = wordDict()
    for i in range(10):
        ch = word_dict.int2word(i)
        print(ch)
        assert i == word_dict.word2int(ch)