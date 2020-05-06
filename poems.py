#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from word_dict import wordDict
from paths import raw_dir, poems_path, check_uptodate
from random import shuffle
from singleton import Singleton
from utils import split_sentences
import os

_corpus_list = ['corpus.txt']


def _gen_poems():
    print("Parsing poems ...")
    word_dict = wordDict()
    with open(poems_path, 'w',encoding='utf-8') as fout:
        for corpus in _corpus_list:
            with open(os.path.join(raw_dir, corpus), 'r',encoding='utf-8') as fin:
                for line in fin.readlines():
                    sentences = split_sentences(line)
                    all_word_in_dict = True
                    for sentence in sentences:
                        sentence=sentence.strip().split()
                        for ch in sentence:
                            if word_dict.word2int(ch) < 0:
                                all_word_in_dict = False
                                break
                        if not all_word_in_dict:
                            break
                    if all_word_in_dict:
                        fout.write('|'.join(sentences) + '\n')
            print("Finished parsing %s." % corpus)


class Poems(Singleton):

    def __init__(self):
        if not check_uptodate(poems_path):
            _gen_poems()
        self.poems = []
        with open(poems_path, 'r',encoding='utf-8') as fin:
            for line in fin.readlines():
                poem=line.strip().split('|')
                temp_poem=[]
                for sentence in poem:
                    sentence=sentence.strip().split(' ')
                    temp_poem.append(sentence)
                self.poems.append(temp_poem)

    def __getitem__(self, index):
        if index < 0 or index >= len(self.poems):
            return None
        return self.poems[index]

    def __len__(self):
        return len(self.poems)

    def __iter__(self):
        return iter(self.poems)

    def shuffle(self):
        shuffle(self.poems)


# For testing purpose.
if __name__ == '__main__':
    poems = Poems()
    for i in range(10):
        print(' '.join(poems[i]))

