#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from word_dict import wordDict
from gensim import models
from numpy.random import uniform
from paths import word2vec_path, check_uptodate
from poems import Poems
from singleton import Singleton
from utils import WORD_VEC_DIM
import numpy as np
import os

#! /usr/bin/env python3
# -*- coding:utf-8 -*-

from word_dict import wordDict
from gensim import models
from numpy.random import uniform
from paths import word2vec_path, check_uptodate
from poems import Poems
from singleton import Singleton
from utils import WORD_VEC_DIM
import numpy as np
import os


def _gen_word2vec():
    print("Generating word2vec model ...")
    word_dict = wordDict()
    poems = Poems()
    poems=[poem[0]+poem[1]+poem[2]+poem[3] for poem in poems]
    print(poems[1])
    model = models.Word2Vec(poems, size = WORD_VEC_DIM, min_count = 1) # 低频词比较多
    embedding = uniform(-1.0, 1.0, [len(word_dict), WORD_VEC_DIM])
    for i, ch in enumerate(word_dict):
        if ch in model:
            embedding[i, :] = model[ch]
    np.save(word2vec_path, embedding)


class word2Vec(Singleton):

    def __init__(self):
        if not check_uptodate(word2vec_path):
            _gen_word2vec()
        self.embedding = np.load(word2vec_path)
        self.word_dict = wordDict()

    def get_embedding(self):
        return self.embedding

    def get_vect(self, ch):
        return self.embedding[self.word_dict.word2int(ch)]

    def get_vects(self, text):
        return np.stack(map(self.get_vect, text)) if len(text) > 0 \
                else np.reshape(np.array([[]]), [0, WORD_VEC_DIM])


# For testing purpose.
if __name__ == '__main__':
    word2vec = word2Vec()


def _gen_word2vec():
    print("Generating word2vec model ...")
    word_dict = wordDict()
    poems = Poems()
    model = models.Word2Vec(poems, size = WORD_VEC_DIM, min_count = 2) # 现在的分词下词频低，保留的所有词
    embedding = uniform(-1.0, 1.0, [len(word_dict), WORD_VEC_DIM])
    for i, ch in enumerate(word_dict):
        if ch in model:
            embedding[i, :] = model[ch]
    np.save(word2vec_path, embedding)


class word2Vec(Singleton):

    def __init__(self):
        if not check_uptodate(word2vec_path):
            _gen_word2vec()
        self.embedding = np.load(word2vec_path)
        self.word_dict = wordDict()

    def get_embedding(self):
        return self.embedding

    def get_vect(self, ch):
        return self.embedding[self.word_dict.word2int(ch)]

    def get_vects(self, text):
        return np.stack(map(self.get_vect, text)) if len(text) > 0 \
                else np.reshape(np.array([[]]), [0, WORD_VEC_DIM])


# For testing purpose.
if __name__ == '__main__':
    word2vec = word2Vec()

