#! /usr/bin/env python3
#-*- coding:utf-8 -*-

from paths import raw_dir, sxhy_path, check_uptodate
from singleton import Singleton
from utils import is_cn_sentence, split_sentences
from typing import List
import os


_rawsxhy_path = os.path.join(raw_dir, 'shixuehanying.txt')


def _gen_sxhy_dict() -> None:
    print("Parsing shixuehanying dictionary ...")
    words = set()
    with open(_rawsxhy_path, 'r',encoding='utf-8') as fin:
        for line in fin.readlines():
            if line[0] == '<':
                continue
            for phrase in line.strip().split()[1:]:
                if not is_cn_sentence(phrase):
                    continue
                idx = 0
                while idx + 4 <= len(phrase):
                    # Cut 2 words each time.
                    words.add(phrase[idx : idx + 2])
                    idx += 2
                # Use jieba to cut the last 3 words.
                if idx < len(phrase):
                    for word in jieba.lcut(phrase[idx:]):
                        words.add(word)
    with open(sxhy_path, 'w') as fout:
        fout.write(' '.join(words))


class Segmenter(Singleton):

    def __init__(self):
        if not check_uptodate(sxhy_path):
            _gen_sxhy_dict()
        with open(sxhy_path, 'r') as fin:
            self.sxhy_dict : Set[str] = set(fin.read().split())

    def segment(self, sentence: str) -> List[str]:
       return sentence  # 都预处理好了 2333

# For testing purpose.
if __name__ == '__main__':
    segmenter = Segmenter()
    with open(r'.\raw\corpus.txt', 'r',encoding='utf-8') as fin:
        for line in fin.readlines()[0 : 6]:
            for sentence in split_sentences(line):
                print(' '.join(segmenter.segment(sentence)))

