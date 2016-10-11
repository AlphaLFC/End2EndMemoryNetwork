# -*- coding: utf-8 -*-
"""
Created on Wed Sep 14 17:49:12 2016

@author: sean
"""
from __future__ import absolute_import
from __future__ import print_function


from sklearn import metrics
from data_utils import DataUtils
from config import Config
import tensorflow as tf
import numpy as np
import pandas as pd
import time, os, random
    
    
class Memory(object):
    
    def __init__(self, config):
        self.config = config
        self.load_data()
        self.add_placeholders()
        self.add_initializer()
        self.add_embedding()
        self.logits, self.in_probs = self.add_model()
        self.loss_op = self.add_loss_op(self.logits)
        self.train_op = self.add_train_op(self.loss_op)    
        self.predict_op = self.add_predict_op(self.logits)
        self.predict_proba_op = self.add_predict_proba_op(self.logits)
        self.init_op = tf.initialize_all_variables()
        self.saver = tf.train.Saver()
    
    def load_data(self):
        self.du = DataUtils(self.config)
        self.embedding_size = self.du.embedding_size
        self.sentence_size = self.du.sentence_size
        self.memory_size = self.du.memory_size
        self.vocab_size = self.du.vocab_size
        self.vocab = self.du.vocab
        self.trainS = self.du.trainS
        self.trainQ = self.du.trainQ
        self.trainA = self.du.trainA
        self.valS = self.du.valS
        self.valQ = self.du.valQ
        self.valA = self.du.valA
        self.testS = self.du.testS
        self.testQ = self.du.testQ
        self.testA = self.du.testA
        self.train_labels = self.du.train_labels
        self.val_labels = self.du.val_labels
        self.test_labels = self.du.test_labels
        self.data_length = len(self.du.data)

    def add_placeholders(self):
        self.stories = tf.placeholder(tf.int32,
            [None, self.memory_size, self.sentence_size], name='stories')
        self.queries = tf.placeholder(tf.int32,
            [None, self.sentence_size], name='queries')
        self.answers = tf.placeholder(tf.int32,
            [None, self.vocab_size], name='answers')
    
    def add_initializer(self, initializer=None):
        if initializer:
            self.init = initializer
        else:
            self.init = tf.random_normal_initializer(
                stddev=self.config.stddev)
    
    def add_embedding(self):
        with tf.variable_scope(self.config.name):
            nil_word_slot = tf.zeros([1, self.embedding_size])
            A_emb = tf.Variable(tf.concat(0, [nil_word_slot, self.init(
                [self.vocab_size, self.embedding_size])]), name='A_emb')
            B_emb = tf.Variable(tf.concat(0, [nil_word_slot, self.init(
                [self.vocab_size, self.embedding_size])]), name='B_emb')
            self.nil_vars = set([A_emb.name, B_emb.name])
            self.m_emb = tf.nn.embedding_lookup(A_emb, self.stories)
            self.q_emb = tf.nn.embedding_lookup(B_emb, self.queries)
    
    def create_feed_dict(self, stories, queries, answers=np.array([])):
        feed = {self.stories: stories, self.queries: queries}
        if answers.any():
            feed[self.answers] = answers
        return feed
    
    def add_model(self):
        with tf.variable_scope(self.config.name):
            b = tf.Variable(self.init([self.memory_size, 
                                        self.embedding_size]), name='b')
            H = tf.Variable(self.init([self.embedding_size, 
                                       self.embedding_size]), name='H')
            W = tf.Variable(self.init([self.embedding_size,
                                       self.vocab_size]), name='W')
            encoding = self.helper_funcs('encoding')
            encoding = encoding(self.sentence_size, self.embedding_size)
            encoding = tf.constant(encoding, name='encoding')
            u = [tf.reduce_sum(self.q_emb * encoding, 1)]
            for hop in range(self.config.hops):
                m = tf.reduce_sum(self.m_emb * encoding, 2) + b
                u_t = tf.transpose(tf.expand_dims(u[-1], -1), [0, 2, 1])
                dotted = tf.reduce_sum(m * u_t, 2)
                probs = tf.nn.softmax(dotted)
                if hop == 0:
                    in_probs = probs
                else:
                    in_probs = tf.concat(0, [in_probs, probs])
                probs_t = tf.transpose(tf.expand_dims(probs, -1), [0, 2, 1])
                c_t = tf.transpose(m, [0, 2, 1])
                o_k = tf.reduce_sum(c_t * probs_t, 2)
                u_k = tf.nn.relu(tf.matmul(u[-1], H) + o_k)
                u.append(u_k)
            return tf.matmul(u_k, W), in_probs

    def add_loss_op(self, logits):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits,
            tf.cast(self.answers, tf.float32), name='cross_entropy')
        loss_op = tf.reduce_sum(cross_entropy, name='loss_op')
        return loss_op
            
    def add_train_op(self, loss_op):
        optimizer = tf.train.AdamOptimizer(epsilon=self.config.epsilon,
            learning_rate=self.config.learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss_op)
        grads_and_vars = [(tf.clip_by_norm(g, self.config.max_grad_norm), v)
                          for g, v in grads_and_vars]
        add_gradient_noise = self.helper_funcs('noise')
        grads_and_vars = [(add_gradient_noise(g), v)
                          for g, v in grads_and_vars]
        nil_grads_and_vars = []
        for g, v in grads_and_vars:
            if v.name in self.nil_vars:
                zero_nil_slot = self.helper_funcs('nil')
                nil_grads_and_vars.append((zero_nil_slot(g), v))
            else:
                nil_grads_and_vars.append((g, v))
        train_op = optimizer.apply_gradients(nil_grads_and_vars, 
                                             name='train_op')
        return train_op

    def add_predict_op(self, logits):
        predict_op = tf.argmax(logits, 1, name='predict_op')
        return predict_op

    def add_predict_proba_op(self, logits):
        predict_proba_op = tf.nn.softmax(logits, name='predict_proba_op')
        return predict_proba_op       
        
    def run_epoch(self, session, loading=False):
        tf.set_random_seed(self.config.random_state)
        if not os.path.exists('./weights'):
            os.makedirs('./weights')
        if loading:
            self.saver.restore(session, self.config.load_file)
            best_loss, best_train_acc, best_val_acc, best_test_acc = \
                self.restore(session)
            best_epoch = 0
            if not os.path.exists(self.config.save_file):
                self.saver.save(session, self.config.save_file)
        else:
            session.run(self.init_op)
            best_loss = float('inf')
        n_train = self.trainS.shape[0]
        batch_size = self.config.batch_size
        batches = zip(range(0, n_train-batch_size, batch_size), 
                      range(batch_size, n_train, batch_size))
        start_time = time.time()
        for i in range(1, self.config.epochs+1):
            np.random.shuffle(batches)
            total_loss = 0.0
            for start, end in batches:
                s = self.trainS[start:end]
                q = self.trainQ[start:end]
                a = self.trainA[start:end]
                feed = self.create_feed_dict(s, q, a)
                loss, _ = session.run([self.loss_op, self.train_op], 
                                      feed_dict=feed)
                total_loss += loss
            if i % self.config.evaluation_interval == 0:
                accuracy = self.helper_funcs('prediction')
                train_acc = accuracy(self.trainS, self.trainQ, 
                    self.train_labels, session, self.predict_op)
                val_acc = accuracy(self.valS, self.valQ, 
                    self.val_labels, session, self.predict_op)
                test_acc = accuracy(self.testS, self.testQ, 
                    self.test_labels, session, self.predict_op)
                print_accuracy = self.helper_funcs('print')
                print_accuracy(total_loss, train_acc, val_acc, 
                               test_acc, i, start_time)
                if best_loss > total_loss:
                    best_loss = total_loss
                    best_train_acc = train_acc 
                    best_val_acc = val_acc 
                    best_test_acc = test_acc
                    best_epoch = i
                    self.saver.save(session, self.config.save_file)
        print_accuracy(best_loss, best_train_acc, best_val_acc, 
                       best_test_acc, best_epoch, start_time)
        output_file = self.helper_funcs('output')
        output_file(best_train_acc, best_val_acc, 
                    best_test_acc, self.config.output_file)
        
    def restore(self, session):
        self.saver.restore(session, self.config.load_file)
        n_train = self.trainS.shape[0]
        batch_size = self.config.batch_size
        batches = zip(range(0, n_train-batch_size, batch_size), 
                      range(batch_size, n_train, batch_size))
        np.random.shuffle(batches)
        total_loss = 0.0
        for start, end in batches:
            s = self.trainS[start:end]
            q = self.trainQ[start:end]
            a = self.trainA[start:end]
            feed = self.create_feed_dict(s, q, a)
            loss = session.run(self.loss_op, 
                               feed_dict=feed)
            total_loss += loss
        accuracy = self.helper_funcs('prediction')
        train_acc = accuracy(self.trainS, self.trainQ, 
                             self.train_labels, session, self.predict_op)
        val_acc = accuracy(self.valS, self.valQ, 
                           self.val_labels, session, self.predict_op)
        test_acc = accuracy(self.testS, self.testQ, 
                            self.test_labels, session, self.predict_op)
        print_accuracy = self.helper_funcs('print')
        print_accuracy(total_loss, train_acc, val_acc, test_acc, 0, None)
        return total_loss, train_acc, val_acc, test_acc
        
    def random_test(self, session, loading=True):
        task_id = random.randint(0, self.data_length)
        task = self.du.data[task_id]
        story = [reduce(lambda x, y: x+' '+y, sent)+'.' for sent in task[0]]
        query = reduce(lambda x, y: x+' '+y, task[1]) + '?'
        answer = task[2][0]
        s, q, a = self.du.vectorize_data([task])
        if loading:
            self.saver.restore(session, self.config.load_file)
        feed = self.create_feed_dict(s, q)
        logits, in_probs = session.run([self.logits, self.in_probs], 
                           feed_dict=feed)
        pred = session.run(self.predict_op, feed_dict=feed)
        prob = session.run(self.predict_proba_op, feed_dict=feed)
        prob = prob[-1][pred[0]]
        pred = self.vocab[pred[0]-1]
    
        if answer == pred:
            result = (pred, prob, 'correct')
        else:
            result = (pred, prob, 'incorrect')
        return story, query, answer, result, in_probs
                
    def helper_funcs(self, key):
        def position_encoding(sentence_size, embedding_size):
            encoding = np.ones((sentence_size, embedding_size),
                               dtype=np.float32)
            for i in range(0, embedding_size):
                for j in range(0, sentence_size):
                    encoding[j, i] = (i+1-embedding_size/2)*\
                                     (j+1-sentence_size/2)
            encoding = 1 + 4 * encoding / embedding_size / sentence_size
            return encoding
        
        def zero_nil_slot(t, name=None):
            with tf.op_scope([t], name, 'zero_nil_slot') as name:
                t = tf.convert_to_tensor(t, name='t')
                s = tf.shape(t)[1]
                z = tf.zeros(tf.pack([1, s]))
                return tf.concat(0, [z, tf.slice(t, [1, 0], [-1, -1])],
                                 name=name)
                                     
        def add_gradient_noise(t, stddev=1e-3, name=None):
            with tf.op_scope([t, stddev], name, 'add_gradient_noise') \
                as name:
                t = tf.convert_to_tensor(t, name='t')
                gn = tf.random_normal(tf.shape(t), stddev=stddev)
                return tf.add(t, gn, name=name)
                
        def predict_accuracy(stories, queries, labels, sess, predict_op,
                             create_feed_dict=self.create_feed_dict):
            length = stories.shape[0]
            results = []
            task_num = len(self.config.task_ids)
            for start in range(0, length, length/task_num):
                end = start + length/task_num
                s = stories[start:end]
                q = queries[start:end]
                feed = create_feed_dict(s, q)
                prediction = sess.run(predict_op, feed_dict=feed)
                accuracy = metrics.accuracy_score(
                    prediction, labels[start:end])
                results.append(accuracy)
            return results
            
        def print_accuracy(total_loss, train_acc, val_acc, 
                           test_acc, epoch, start_time):
            print('====================================')
            print('Epoch: ', epoch)
            print('Total loss: {}'.format(int(total_loss)))
            if start_time:
                print('Total time cost: {}'\
                      .format(int(time.time()-start_time)))
            print()
            t = 1
            for t1, t2, t3 in zip(train_acc, val_acc, test_acc):
                print('Task {}'.format(self.config.task_ids[t-1]))
                print("Training Accuracy = {:.2}".format(t1))
                print("Validation Accuracy = {:.2}".format(t2))
                print("Testing Accuracy = {:.2}".format(t3))
                print()
                t += 1
            print('====================================')
            
        def output_file(train_acc, val_acc, test_acc, filename):
            print('Writing results to {}'.format(self.config.output_file))
            df = pd.DataFrame({
            'Training Accuracy': train_acc,
            'Validation Accuracy': val_acc,
            'Testing Accuracy': test_acc
            }, index=self.config.task_ids)
            df.index.name = 'Task'
            df.to_csv(filename)
            
        f_dict = {'encoding': position_encoding, 
                  'nil': zero_nil_slot,
                  'noise': add_gradient_noise,
                  'prediction': predict_accuracy,
                  'print': print_accuracy,
                  'output': output_file}
        return f_dict[key]


if __name__ == '__main__':
    config = Config()
    model = Memory(config)
    session = tf.Session()
    model.run_epoch(session)
    story, query, answer, result, in_probs = model.random_test(session)
    for s in story:
        print(s)
    print()
    print('Question: ', query)
    print('Predicted answer: {}, Probability: {:.2}, {}'.format(
        result[0], result[1], result[2]))
    print('Correct answer: {}'.format(answer))
    print(in_probs)
