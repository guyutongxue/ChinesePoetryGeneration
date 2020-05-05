'''
This file generate raw_dict.pkl from corpus.txt.
raw_dict is a dict which stores the occurance time of each word.
'''

import pickle as pkl
storefile2 = r'raw/raw_dict.pkl'
rawfile = r'raw/corpus.txt'
words = dict()
with open(rawfile, 'r', encoding='utf-8') as fin:
    lines = fin.readlines()
    size = len(lines)
    l = 0
    for line in lines:
        sentences = line.split('|')
        for sentence in sentences:
            wordlist = sentence.split(' ')
            for word in wordlist:
                word = word.strip()
                words[word] = words.get(word, 0) + 1
        l += 1
        if l % 100 == 0:
            print(l / size)
        if l < 100:
            print(words)
pkl.dump(words, open(storefile2, 'wb'))
