# encoding=utf-8
import logging
import gensim
import sys
import time
import os
import cPickle
import pickle
import jieba
import re
import lda
import numpy as np
from gensim import interfaces, utils, matutils
import util
from LDA import SklearnBasedLda
from LDA import GensimBasedLda
reload(sys)
sys.path.append('..')
import utilities.utilities as utilities
sys.setdefaultencoding('utf-8')
logging.basicConfig(format = '%(asctime)s %(levelname)s : %(message)s', level = logging.INFO)

def calc_perplexity():
    test_corpus = '../data/corpus/combine_corpus_filter_again_v2_test.txt'
    processed_docs = util.read_corpus_and_preprocess(test_corpus)
    corpus = [id_word_dict.doc2bow(pdoc) for pdoc in processed_docs]

    #model = LdaBasedLda()
    model = GensimBasedLda()
    model.load_model(model_file, dict_file)
    perplexity = model.model.log_perplexity(corpus)
    print perplexity

def topic_dist():
    model_file = '../model/LDA/topic150_filter_again_v2/LDA.model'
    corpus_file = '../data/aux/dict_no_below_20_no_above_05_again_v2'
    model = GensimBasedLda()
    model.load_model(model_file, corpus_file)

    doc_file = '../data/corpus/pkbigdata_title_content_corpus.txt'
    doc_topic = []
    with open(doc_file) as f:
        lines = f.readlines()
        for i in range(len(lines))[::2]:

            if i % 1000 == 0:
                print i
            query = lines[i]+lines[i+1]
            vec = model.get_topic_distribution(jieba.cut(query, cut_all=False))
            doc_topic.append(vec)

    vec_file = './doc_topic_150'
    pickle.dump(doc_topic, open(vec_file,'w'))

    # vec_file = './doc_topic'
    # vec = pickle.load(file(vec_file))
    # print vec[1:5]
    # print len(vec)

model_file = '../model/LDA/model_by_gensim/iter_100_topics_75_1502806378/LDA.model'
dict_file = '../data/aux/dict_no_below_20_no_above_05_1502791341'
if __name__ == '__main__':
    '''
    doc_file = '../data/corpus/combine_corpus_filter_again_v2_sub.txt'
    dict_file = '../data/aux/dict_no_below_20_no_above_05_1502791341'
    test_corpus = '../data/corpus/combine_corpus_filter_again_v2_test.txt'

    processed_docs, id_word_dict, word_id_dict, vocab = util.doc2dict(doc_file, dict_file)
    #doc_word_chunk = util.docs_to_matrix(processed_docs, word_id_dict)

    # model = SklearnBasedLda()
    # model.train(vocab, doc_word_chunk, 75, 2000)
    # model.show_topics()

    model = GensimBasedLda()
    model.train(processed_docs, id_word_dict, 100, 200)
    #model = LdaBasedLda()
    #model.load('model/LDA/model_by_lda/iter_2000_topics_75_1502815418')
    #topic_dist = model.topic_word_

    #calc_perplexity()
    '''
    topic_dist()
