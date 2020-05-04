#! /usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Execute the TextRank algorithm.

"""

from paths import raw_dir, sxhy_path, wordrank_path, check_uptodate
from poems import Poems
from utils import is_cn_sentence, split_sentences
from segment import Segmenter
from singleton import Singleton
from typing import Dict, List, Tuple, Set
import json
import os
import sys
import jieba


_stopwords_path = os.path.join(raw_dir, 'stopwords.txt')

_rawsxhy_path = os.path.join(raw_dir, 'shixuehanying.txt')


def _gen_sxhy_dict() -> None:
    """
    Generate word-seperated version of *ShixueHanying* dictionary from its raw text. 

    The generated file will be saved at `sxhy_path`.
    """
    print("Parsing shixuehanying dictionary ...")
    words: Set[str] = set()
    with open(_rawsxhy_path, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            if line[0] == '<':
                continue
            for phrase in line.strip().split()[1:]:
                if not is_cn_sentence(phrase):
                    continue
                idx: int = 0
                while idx + 4 <= len(phrase):
                    # Cut 2 chars each time.
                    words.add(phrase[idx: idx + 2])
                    idx += 2
                # Use jieba to cut the last 3 chars.
                if idx < len(phrase):
                    for word in jieba.lcut(phrase[idx:]):
                        words.add(word)
    with open(sxhy_path, 'w', encoding='utf-8') as fout:
        fout.write(' '.join(words))


def _get_stopwords() -> Set[str]:
    """
    Get stopwords from `stopwords.txt`. Returns a set including all stopwords.
    """
    stopwords: Set[str] = set()
    with open(_stopwords_path, 'r', encoding='utf-8') as fin:
        for line in fin.readlines():
            stopwords.add(line.strip())
    return stopwords

# 阻尼系数
_DAMP: float = 0.85

class RankedWords(Singleton):

    def __init__(self):
        # 获取停用词
        self.stopwords = _get_stopwords()

        # 生成分词的《诗学含英》
        if not check_uptodate(sxhy_path):
            _gen_sxhy_dict()
        with open(sxhy_path, 'r') as fin:
            self.sxhy_dict: Set[str] = set(fin.read().split())

        # 生成 TextRank
        if not check_uptodate(wordrank_path):
            self._do_text_rank()
        with open(wordrank_path, 'r', encoding='utf-8') as fin:
            self.word_scores: List[Tuple[str, float]] = json.load(fin)
        self.rank: Dict[str, int] = dict(
            (ws[0], i) for i, ws in enumerate(self.word_scores))

    def _do_text_rank(self) -> None:
        """
        Execute the TextRank algorithm.
        """

        print("Do text ranking ...")
        # 获取共现矩阵
        adjlists = self._get_adjlists()
        print("[TextRank] Total words: %d" % len(adjlists))

        # 执行 TextRank 算法。
        # 类似于 PageRank 算法。不断地执行
        # $$S(V_i):=(1-d)+d\cdot\sum_{j=1}^{|V|}r_{ij}S(V_j)$$
        # 来更新 S(V_i) 直至稳定，就得到了每个 V_i 的权重。
        # 
        # 使用字典来表示 S(V_i) 这样的结构。通过 newscores 和 oldsscores
        # 两个字典交替使用、更新，直至稳定。下面是这个字典存放的数据示意：
        # *scores= dict {
        #     "word1" : 1.0,
        #     "word2" : 1.0,
        #     ...
        # }
        # 其中 "word*" 指需要计算的单词 V_i，对应的 1.0 即权重 S(V_i)。

        newscores: Dict[str, int] = {}
        oldscores: Dict[str, int] = {}
        for i in adjlists:
            oldscores[i] = 1.0
        
        # 迭代数
        itr = 0
        while True:
            sys.stdout.write("[TextRank] Iteration %d ..." % itr)
            sys.stdout.flush()
            for i, subdict in adjlists.items():
                newscores[i] = (1.0 - _DAMP) + _DAMP * \
                    sum(adjlists[j][i] * oldscores[j] for j in subdict)
            eps = 0
            for word in newscores:
                eps = max(eps, abs(oldscores[word] - newscores[word]))
                oldscores[word] = newscores[word]
            print(" eps = %f" % eps)
            if eps <= 1e-6:
                break
            itr += 1


        def cmp_key(x: Tuple[str, int]) -> Tuple[int, int]:
            
            word, score = x
            return (0 if word in self.sxhy_dict else 1, -score)

        # 将所得到的 word-rank 的 dict 进行排序，整理为 word-rank 的 list
        # 排序方式为：出自《诗学含英》的优先排在列表头。其次 rank 高的优先排在列表头。

        # 下方排序中的 key Lambda: 作用于 iterable 中的每一个元素
        # 并将函数返回的结果作为比较对象。这里返回的是二元元组
        # `(x,y)`
        # 其中 x 表明在《诗学含英》中的出现， y 表明计算得到的 Rank。
        # 由于 sorted 是小序在首，故若出现记 x = 0；Rank 取负数。

        words: List[Tuple[str, float]] = sorted(
            [(word, score) for word, score in newscores.items()],
            key=lambda x: (0 if x[0] in self.sxhy_dict else 1, -x[1])
        )

        # Store ranked words and scores.
        with open(wordrank_path, 'w', encoding='utf-8') as fout:
            json.dump(words, fout)

    def _get_adjlists(self) -> Dict[str, Dict[str, float]]:
        print("[TextRank] Generating word graph ...")
        segmenter = Segmenter()
        poems = Poems()
        
        # 获取共现矩阵（邻接表）：两个词在同一句中共同出现的次数。使用 adjlists
        # 来存储。下面展示了它的示例：
        # adjlists= dict {
        #     "word1" : dict {
        #         "word2" : 1.0,
        #         "word3" : 1.0,
        #         ...
        #     },
        #     "word2" : dict {
        #         "word1" : 1.0,
        #         "word3" : 1.0,
        #         ...
        #     }
        #     ...
        # }
        # 其中 "word*" 指明单词 V_i, V_j；对应的 1.0 表示“边权值” w_{ij}。
        # 在这个阶段，矩阵是对称的，即 w_{ij}=w{ji}。

        adjlists: Dict[str, Dict[str, float]] = dict()
        for poem in poems:
            for sentence in poem:
                words: List[str] = []
                for word in segmenter.segment(sentence):
                    if word not in self.stopwords:
                        words.append(word)
                for word in words:
                    if word not in adjlists:
                        adjlists[word] = dict()
                for _, i in enumerate(words):
                    for j in words[_+1:]:
                        if j not in adjlists[i]:
                            adjlists[i][j] = 1.0
                        else:
                            adjlists[i][j] += 1.0
                        if i not in adjlists[j]:
                            adjlists[j][i] = 1.0
                        else:
                            adjlists[j][i] += 1.0
                            
        # 对矩阵 W 进行预处理（正规化）
        # $$r_{ji}=\frac{w_{ij}}{\displaystyle\sum_{k=1}^{|V|}w_{jk}}$$

        for j in adjlists:
            sum_k = sum(k for k in adjlists[j].values())
            for i in adjlists[j]:
                adjlists[j][i] /= sum_k
        return adjlists

    def __getitem__(self, index: int) -> float:
        """ `operator[]` in C++"""
        if index < 0 or index >= len(self.word_scores):
            return None
        return self.word_scores[index][0]

    def __len__(self) -> int:
        """ defines when using `len(...)` what should it do"""
        return len(self.rank)

    def __iter__(self):
        """ defines when using `iter(...)` (may be used in for-range) what should it do"""
        return map(lambda x: x[0], self.word_scores)

    def __contains__(self, word: str) -> bool:
        """defines when using `in` what should it do"""
        return word in self.rank

    def get_rank(self, word: str) -> int:
        if word not in self.rank:
            return len(self.rank)
        return self.rank[word]


# 测试用
if __name__ == '__main__':
    ranked_words = RankedWords()
    for i in range(100):
        print(ranked_words[i])
