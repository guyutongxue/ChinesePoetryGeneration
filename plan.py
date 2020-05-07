# -*- coding:utf-8 -*-
"""
对于输入的首句，得到三个句子中心词
"""
from data_utils import gen_train_data
from gensim import models
from paths import save_dir, plan_data_path, check_uptodate
from random import random, shuffle, randint
from rank_words import RankedWords
from singleton import Singleton
import jieba
import os

_plan_model_path = os.path.join(save_dir, 'plan_model.bin')
NUM_OF_SENTENCES = 3


def is_cn_char(ch):
    """ 判断是否为中文字符 """
    return ch >= u'\u4e00' and ch <= u'\u9fa5'


def split_sentences(text):
    """把输入文本划分为短句的列表"""
    sentences = []
    i = 0
    for j in range(len(text) + 1):
        if j == len(text) or text[j] in [u'，', u'。', u'！', u'？', u'、', u'\n', u' ']:
            if i < j:
                sentence = u''.join(filter(is_cn_char, text[i:j]))
                sentences.append(sentence)
            i = j + 1
    return sentences


def train_planner():
    """利用gensim,将提取的关键词向量化"""
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    if not check_uptodate(plan_data_path):
        gen_train_data()
    keywords_list = []  # 格式：[ ['keyword1', 'keyword2', 'keyword3', 'keyword4'] ]
    with open(plan_data_path, 'r') as infile:
        for line in infile.readlines():
            keywords_list.append(line.strip().split('\t'))
    # word2vec 训练词向量
    model = models.Word2Vec(keywords_list, size=512, window=4, min_count=5)
    model.save(_plan_model_path)


class Planner(Singleton):
    """planner"""

    def __init__(self):
        """"初始化：加载rankedwords; 关键词向量化; 得到rankedwords到word_idx的dict"""
        self.ranked_words = RankedWords()
        if not os.path.exists(_plan_model_path):
            train_planner()
        self.model = models.Word2Vec.load(_plan_model_path)
        self.word2idx = {}
        for idx, item in enumerate(self.ranked_words):
            self.word2idx[item] = idx

    def extract(self, text):
        """利用rankedwords,从输入文本中提取关键词"""
        keywords = set()
        for sentence in split_sentences(text):
            # 过滤器：过滤出既在text中，又在ranked_words中的词语
            keywords.update(
                filter(lambda w: w in self.ranked_words, jieba.lcut(sentence)))
        return keywords

    def plan(self, text):
        """得到四个关键词"""
        keywords = self.extract(text)
        result=[]
        filtered_keywords = list(
            filter(lambda w: w in self.model.wv, keywords))
        if len(filtered_keywords) > 0:  # 存在在model中的keywords
            # 寻找与model中的keywords中的词最相似的词语
            for i in range(NUM_OF_SENTENCES):
                similars = self.model.wv.most_similar(
                    positive=filtered_keywords)
                similars = sorted(similars, key=lambda x: x[1])
                result.append(similars[0][0])
                filtered_keywords.append(similars[0][0])
        else:
            # 否则只好随机找一个关键词，再生成剩余关键词
            while True:
                idx = randint(0, len(self.ranked_words) - 1)
                if self.ranked_words[idx] in self.model:
                    break
            result.append(self.ranked_words[idx])
            filtered_keywords.append(self.ranked_words[idx])
            for i in range(NUM_OF_SENTENCES - 1):
                similars = self.model.wv.most_similar(
                    positive=filtered_keywords)
                similars = sorted(similars, key=lambda x: x[1])
                result.append(similars[0][0])
                filtered_keywords.append(similars[0][0])

        return result


# For testing purpose.
if __name__ == '__main__':
    planner = Planner()
    keywords = planner.plan("横看成岭侧成峰")
    print(keywords)
