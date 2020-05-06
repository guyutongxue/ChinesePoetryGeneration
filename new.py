# -*- coding:utf-8 -*-

import numpy as np
import os
import sys
import tensorflow as tf


BATCH_SIZE = 128
NUM_UNITS = 512
CHAR_VEC_DIM = 512


class Generator():
    def __init__(self):  #构造函数
        self.char_dict = None  #字典长度,之后调用
        #char2vec
        self._build_layers()  #生成结构
        self.saver = tf.train.Saver(tf.global_variables())  #定义模型

    '''def _session_initializer(self):'''

    def _build_layers(self):#构造函数中调用
        self._build_keyword_encoder()
        self._build_context_encoder()
        #self._build_decoder()
        self._build_softmax()
        self._build_optimizer()

    def _build_keyword_encoder(self):
        self.keyword = tf.placeholder(
            shape=[BATCH_SIZE, None, CHAR_VEC_DIM],
            dtype=tf.float32,
            name="keyword")#keyword
        self.keyword_length = tf.placeholder(shape=[BATCH_SIZE],
                                                dtype=tf.int32,
                                                name="keyword_length")#keyword长度                                   
        outputs, states, = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=tf.contrib.rnn.GRUCell(num_units=NUM_UNITS / 2),
            cell_bw=tf.contrib.rnn.GRUCell(num_units=NUM_UNITS / 2),
            inputs=self.keyword,
            sequence_length=self.keyword_length,
            dtype=tf.float32,
            scope="keyword_encoder")
        self.keyword_state = tf.concat(states, axis=1)#concat双向的input 
        self.keyword_outputs = tf.concat(outputs, axis=2)

    def _build_context_encoder(self):
        self.context = tf.placeholder(
            shape=[BATCH_SIZE, None, CHAR_VEC_DIM],
            dtype=tf.float32,
            name="context")
        self.context_length = tf.placeholder(shape=[BATCH_SIZE],
                                                dtype=tf.int32,
                                                name="context_length")                                   
        outputs, states = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=tf.contrib.rnn.GRUCell(num_units=NUM_UNITS / 2),
            cell_bw=tf.contrib.rnn.GRUCell(num_units=NUM_UNITS / 2),
            inputs=self.context,
            sequence_length=self.context_length,
            dtype=tf.float32,
            scope="context_encoder")
        self.context_state = tf.concat(states, axis=1)
        self.context_outputs = tf.concat(outputs, axis=2)

    def _build_decoder(self):
        self.decoder_inputs = tf.placeholder(
                shape = [BATCH_SIZE, None, CHAR_VEC_DIM],
                dtype = tf.float32, 
                name = "decoder_inputs")
        self.decoder_input_length = tf.placeholder(
                shape = [BATCH_SIZE],
                dtype = tf.int32,
                name = "decoder_input_length")

        
        attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
            num_units = NUM_UNITS,
            memory = self.context_outputs,#memory为encoder的输出
            memory_sequence_length = self.context_length,
            name = "BahdanauAttention"
        )
        basicGRUcell = tf.contrib.rnn.GRUCell(NUM_UNITS)
        decoder_rnn_cell = tf.contrib.seq2seq.AttentionWrapper(
            cell = basicGRUcell,
            attention_mechanism = attention_mechanism,
            output_attention=True,
            name = "decoder_rnn_cell"
        )

        self.initial_state = decoder_rnn_cell.zero_state(
                batch_size = BATCH_SIZE, dtype = tf.float32).clone(cell_state = tf.concat([self.keyword_state,self.context_state]))


        outputs, state = tf.nn.dynamic_rnn(
                cell = decoder_rnn_cell,
                inputs = self.decoder_inputs,
                sequence_length = self.decoder_input_length,
                initial_state = self.initial_state,
                dtype = tf.float32, 
                scope = "decoder_rnn")

        self.decoder_outputs = outputs
        self.decoder_state = state

    def _reshape_decoder_outputs(self):
        

    def _build_softmax(self):
        t = tf.truncated_normal_initializer(stddev=0.08)#(-0.08, 0.08)论文
        weight = tf.get_variable("weight",
                                 [NUM_UNITS, len(self.char_dict)],
                                 initializer=t)
        bias = tf.get_variable("bias", [len(self.char_dict)], initializer=t)
        reshaped_outputs = self._reshape_decoder_outputs()#还没写
        self.logits = tf.add(tf.matmul(reshaped_outputs,weight), bias)#全连接层不需要激活函数
        self.probs = tf.nn.softmax(self.logits)


    def _build_optimizer(self):
        lr = 1e-3
        self.labels_placeholder = tf.placeholder(shape=[None],
                                                 dtype=tf.int32,
                                                 name="labels_placeholder")
        num_classes = len(self.char_dict)
        labels = tf.one_hot(self.labels_placeholder, num_classes)
        losses = tf.nn.softmax_cross_entropy_with_logits(labels=labels,
                                                         logits=self.logits)
        self.mean_loss = tf.reduce_mean(losses)
        self.train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(
            self.mean_loss)

    def gen_prob_list(self, probs, context, pron_dict):#param probs:softmax输出的概率分布 context:已生成的句子 pron_dict:音律字典
        probs_list = probs.tolist()#array->list 取首行
        prob_list = probs_list[0]
        prob_list[0] = 0
        prob_list[-1] = 0#首尾置0
        idx = len(context)
        used_chars = set(ch for ch in context)
        for i in range(1, len(prob_list) - 1):
            word = self.char_dict.int2char(i)
            word_length = len(word)
            ch = word[-1]#末尾单字
            #idx比list下标要大1(因为有start_of_sentence为context[0])
            #一行7个字一个end e.g:a[1]-a[7]=char a[8]=$
            # Penalize used characters. 字词重复
            if ch in used_chars:
                prob_list[i] *= 0.6#最好再降低一些

            #超字处理，默认词语最长为2，直接概率清零，确保不超
            if(idx == 7 or idx == 15 or idx == 23 or idx == 31) and (word_length == 2):
                prob_list[i] *= 0
                continue
            #下面的押韵、平仄同理可以后面再细改，等确定数据形式我再来改
            if (idx == 16-word_length or idx == 32-word_length) and \
                    not pron_dict.co_rhyme(ch, context[7]):
                prob_list[i] *= 0.2#不押韵可以继续降低权重，保证视觉效果

            if idx > 2 and idx % 8 == 2 and \
                    not pron_dict.counter_tone(context[2], ch) and word_length == 1:
                prob_list[i] *= 0.4
            if idx > 2 and idx % 8 == 1 and \
                    not pron_dict.counter_tone(context[2], ch) and word_length == 2:
                prob_list[i] *= 0.4

            if (idx % 8  == 4 or idx % 8 == 6) and \
                    not pron_dict.counter_tone(context[idx - 2], ch) and word_length == 1:
                prob_list[i] *= 0.4
            if (idx % 8  == 3 or idx % 8 == 5) and \
                    not pron_dict.counter_tone(context[idx - 2], ch) and word_length == 2:
                prob_list[i] *= 0.4
        return prob_list

    