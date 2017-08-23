# encoding=utf-8
import logging
import gensim
import sys
import time
import math
import os
import cPickle
import pickle
import jieba
import re
import lda
import numpy as np
from gensim import interfaces, utils, matutils
import logging
reload(sys)
sys.path.append('..')
import utilities.utilities as utilities
sys.setdefaultencoding('utf-8')
logging.basicConfig(level = logging.DEBUG, format = '%(asctime)s %(levelname)s: %(message)s')

def read_corpus_and_preprocess(doc_file):
    stop_words = utilities.read_stopwords('../data/aux/stop_words')
    processed_docs = []
    with open(doc_file, 'r') as f:
        lines = f.readlines()
        for i in range(0,len(lines),2):
            title_and_content = lines[i]+lines[i+1]
            words = re.split(' |\n', title_and_content)
            words_list = []
            for word in words:
                if not stop_words.has_key(word):
                    words_list.append(word)
            #每一篇文章是一个list
            processed_docs.append(words_list)
    return processed_docs

def doc2dict(doc_file = '', id_word_dict_file = '', no_below = 20, no_above = 0.5):
    """form bow from docs
     Args:
         doc_file:
             docs file location
        id_word_dict_file:
         no_below:
             filter words shown less than no_below number docs
         no_above:
             filter words shown more than no_above number docs(in percentage rate)

    Returns:
        processed_docs:
            a list of list, every list contains the words of a documents after preprocess
        id_word_dict:
            a bow dictionary '{id: word}' for each word
        word_id_dict:
            a bow dictionary '{word: id}' for each word
        vocab: tuple
            like ('apple', 'bear', 'cat') to show vocabulary

    """

    stop_words = utilities.read_stopwords('../data/aux/stop_words')
    processed_docs = []
    with open(doc_file, 'r') as f:
        lines = f.readlines()
        for i in range(0,len(lines),2):
            title_and_content = lines[i]+lines[i+1]
            words = re.split(' |\n', title_and_content)
            words_list = []
            for word in words:
                if not stop_words.has_key(word):
                    words_list.append(word)
            #每一篇文章是一个list
            processed_docs.append(words_list)
        # normalized words and their integer ids, a dictionary {id: word}
    if id_word_dict_file == '':
            id_word_dict = gensim.corpora.Dictionary(processed_docs)
            id_word_dict.filter_extremes(no_below = no_below, no_above = no_above)
            timestamp = str(int(time.time()))
            id_word_dict.save('../data/aux/dict_no_below_20_no_above_05_'+timestamp)
    else:
        id_word_dict = gensim.corpora.Dictionary.load(id_word_dict_file)
    logging.info('id_word_dict load/calculate finished!')
    logging.info(str(len(id_word_dict))+' words in the dictionary')
    word_id_dict = {}
    vocab_list = []
    for id in range(0, len(id_word_dict)):
        word = id_word_dict[id]
        word_id_dict[word] = id
        vocab_list.append(word)
    vocab = tuple(vocab_list)
    pickle.dump(word_id_dict, open('word_id_dict','w'))
    logging.info('doc2dict function finished!')
    return processed_docs, id_word_dict, word_id_dict, vocab


def docs_to_matrix(processed_docs, word_id_dict, chunk_size = 200):
    """transform processed_docs to doc-word matrix

    Args:
        inference with doc2dict()
    Returns:
        chunks of doc_word matrix
    """
    docs_num = len(processed_docs)
    words_num = len(word_id_dict)

    doc_word_chunk = []
    doc_word =np.zeros((chunk_size, words_num ), dtype = np.intc)
    for i in range(docs_num):
        if i % 1000 == 0:
            logging.info('processing '+str(i)+' docs')
        doc_index = i % chunk_size
        if doc_index == 0:
            doc_word_chunk.append(doc_word)
            del doc_word
            doc_word =np.zeros((chunk_size, words_num ), dtype = np.intc)

        for word in processed_docs[i]:
            word_u = word.decode('utf-8')

            if word_id_dict.has_key(word_u):
                word_index = word_id_dict[word_u]
                doc_word[doc_index, word_index] += 1
    return doc_word_chunk
