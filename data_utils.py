# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 15:27:12 2016

@author: sean
"""
from __future__ import absolute_import
from __future__ import print_function

from sklearn import cross_validation
from itertools import chain
from six.moves import range, reduce
import re, os
import numpy as np


class DataUtils(object):

    def __init__(self, config):
        self.config = config
        self.embedding_size = self.config.embedding_size
        self.train, self.test = self.load_file()
        self.data = self.get_data()
        self.vocab = self.get_vocab()
        self.word_idx = self.get_word_idx()
        self.vocab_size = len(self.word_idx) + 1
        self.memory_size = self.get_memory_size()
        self.sentence_size = self.get_sentence_size()
        self.trainS, self.valS, self.trainQ, self.valQ, self.trainA,\
            self.valA = self.get_train_val(self.config.val_size)
        self.testS, self.testQ, self.testA = self.get_test()
        self.train_labels, self.test_labels,\
            self.val_labels = self.get_labels()        

    def load_file(self):
        train, test = [], []
        for i in self.config.task_ids:
            tr, te = self.load_task(self.config.data_dir, i)
            train.append(tr)
            test.append(te)
        return train, test

    def get_data(self):
        data = list(chain.from_iterable(self.train + self.test))
        return data

    def get_vocab(self):
        all_words = [set(list(chain.from_iterable(s))+q+a) \
                     for s, q, a in self.data]
        vocab = sorted((reduce(lambda x, y: x.union(y), all_words)))
        return vocab

    def get_word_idx(self):
        word_idx = dict((w, i+1) for i, w in enumerate(self.vocab))
        return word_idx
    
    def get_memory_size(self):
        max_story_size = max(map(len, [s for s, _, _ in self.data]))
        memory_size = min(self.config.memory_size, max_story_size)
        return memory_size
        
    def get_sentence_size(self):
        sentence_size = max(map(len, chain.from_iterable(
            s for s, _, _ in self.data)))
        query_size = max(map(len, [q for _, q, _ in self.data]))
        return max(sentence_size, query_size)

    def get_train_val(self, val_size):
        trainS, valS, trainQ, valQ, trainA, valA = [], [], [], [], [], []
        rs = self.config.random_state
        for task in self.train:           
            S, Q, A = self.vectorize_data(task)
            ts,vs, tq, vq, ta, va = cross_validation.train_test_split(
                S, Q, A, test_size=val_size,random_state=rs)
            trainS.append(ts)
            trainQ.append(tq)
            trainA.append(ta)
            valS.append(vs)
            valQ.append(vq)
            valA.append(va)
        def stack(data):
            return reduce(lambda x, y: np.vstack((x, y)), (z for z in data))
        trainS = stack(trainS)
        trainQ = stack(trainQ)
        trainA = stack(trainA)
        valS = stack(valS)
        valQ = stack(valQ)
        valA = stack(valA)
        return trainS, valS, trainQ, valQ, trainA, valA

    def get_test(self):
        S, Q, A = self.vectorize_data(chain.from_iterable(self.test))
        return S, Q, A
        
    def get_labels(self):
        train_labels = np.argmax(self.trainA, axis=1)
        test_labels = np.argmax(self.testA, axis=1)
        val_labels = np.argmax(self.valA, axis=1)
        return train_labels, test_labels, val_labels
        
    def load_task(self, data_dir, task_id, only_supporting=False):
        assert 0 < task_id < 21
        ss = 'qa{}_'.format(task_id)
        files = os.listdir(data_dir)
        train_file = [os.path.join(data_dir, f) \
                      for f in files if ss in f and 'train' in f][0]
        test_file = [os.path.join(data_dir, f) \
                     for f in files if ss in f and 'test' in f][0]
        train_data = self.get_stories(train_file, only_supporting)
        test_data = self.get_stories(test_file, only_supporting)
        return train_data, test_data

    def tokenize(self, sentence):
        return [w.strip() for w in re.split('(\w+)?', sentence) if w.strip()]

    def parse_stories(self, lines, only_supporting=False):
        story, data = [], []
        for line in lines:
            line = line.lower()
            idx, line = line.split(' ', 1)
            idx = int(idx)
            if idx == 1:
                story = []
            if '\t' in line:
                query, ans, support = line.split('\t')
                ans = [ans]
                query = self.tokenize(query)
                if query[-1] == '?':
                    query = query[:-1]
                if only_supporting:
                    support = map(int, support.split())
                    substory = [story[i-1] for i in support]
                else:
                    substory = [s for s in story if s]
                story.append('')
                data.append((substory, query, ans))
            else:
                sentence = self.tokenize(line)
                if sentence[-1] == '.':
                    sentence = sentence[:-1]
                story.append(sentence)
        return data

    def get_stories(self, filename, only_supporting=False):
        with open(filename) as f:
            return self.parse_stories(f.readlines(), only_supporting)

    def vectorize_data(self, data):
        S, Q, A = [], [], []
        for story, query, ans in data:
            ss = []
            for i, sentence in enumerate(story, 1):
                ls = max(0, self.sentence_size-len(sentence))
                ss.append([self.word_idx[w] for w in sentence] + ls * [0])
            ss = ss[::-1][:self.memory_size][::-1]
            lm = max(0, self.memory_size-len(ss))
            for _ in range(lm):
                ss.append(self.sentence_size*[0])
            lq = max(0, self.sentence_size-len(query))
            q = [self.word_idx[w] for w in query] + lq * [0]
            y = np.zeros(len(self.word_idx)+1)
            for a in ans:
                y[self.word_idx[a]] = 1
            S.append(ss)
            Q.append(q)
            A.append(y)
        return np.array(S), np.array(Q), np.array(A)
        