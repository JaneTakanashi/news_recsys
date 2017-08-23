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
import util
import numpy as np
from gensim import interfaces, utils, matutils
import util
from sklearn.decomposition import LatentDirichletAllocation
reload(sys)
sys.path.append('..')
import utilities.utilities as utilities
sys.setdefaultencoding('utf-8')
logging.basicConfig(format = '%(asctime)s %(levelname)s : %(message)s', level = logging.INFO)
#logging.root.level = logging.INFO

stop_words = {}
processed_docs = []

timestamp = str(int(time.time()))
model_path='../model/LDA'

class GensimBasedLda:

    def get_topic_distribution(self, query_word):
        query_bow =  self.dict.doc2bow(query_word)
        gamma, _ = self.model.inference([query_bow])
        return gamma[0] / sum(gamma[0])

    def predict_topic_dist(self, query_word):
        """show the relative topics of the given words list

        Args:
            query_bow:
                a word list stand for the query
        """

        query_bow = self.dict.doc2bow(query_word)
        return self.model[query_bow]

        # for index, score in sorted(self.model[query_bow], key = lambda tup : -1 * tup[1]):
        #     print "topic {} {} {}".format(index, score, self.model.print_topic(index, 10))

    def train(self, processed_docs, id_word_dict, n_topics, n_iter):
        """
        Args:
            id_word_dict:
                a bow dictionary '{id: word}' for each word

            n_topics: int
                Number of topics

            n_iter: int, default 2000
                Number of sampling iterations

        """
        #doc2bow: converts a collection of words to its bag-og-words repretation: a list of(word_id, word_freqency)2-tuples
        self.n_topics = n_topics

        bag_of_words_corpus = [id_word_dict.doc2bow(pdoc) for pdoc in processed_docs]
        self.dict = id_word_dict
        logging.info('begin to trian model...')

        lda_model = gensim.models.LdaModel(bag_of_words_corpus, num_topics=n_topics, id2word = id_word_dict, iterations = n_iter)
        save_path = os.path.join(model_path,'model_by_genism', 'iter_'+str(n_iter)+'_topics_'+str(n_topics)+'_'+timestamp)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        lda_model.save(os.path.join(save_path, 'LDA.model'))
        self.model = lda_model
        print('topic num {} model traing process finished'.format(n_topics))

    def load_model(self, model_file, dict_file):
        logging.info('begin to load lda model...')
        if not os.path.isfile(model_file):
            logging.error('lda model file doesn\'t exits')
            return
        if not os.path.isfile(dict_file):
            logging.error('lda dictionary file doesn\'t exits')
            return
        self.model = gensim.models.LdaModel.load(model_file)
        self.dict = gensim.corpora.Dictionary.load(dict_file)
        logging.info('lda model has been loaded successfully!')


    def show_topics(self, n_top_words = 10):
        self.model.show_topics(self.n_topic, n_top_words)

    def evaluate(model_file, dict_file, num_topics):
        model = gensim.models.LdaModel.load(model_file)
        model.print_topics(num_topics)
        dictionary = gensim.corpora.Dictionary.load(dict_file)
        num_words = len(dictionary)
        print model.a
        print num_words
        topic_vec = np.zeros((num_topics, num_words), dtype = np.float)

    def show_topics(model, num_topics=10, num_words=10, log=False, formatted=True):
        if num_topics < 0 or num_topics >= model.num_topics:
            num_topics = model.num_topics
            chosen_topics = range(num_topics)
        else:
            num_topics = min(num_topics, model.num_topics)

            # add a little random jitter, to randomize results around the same alpha
            sort_alpha = model.alpha + 0.0001 * model.random_state.rand(len(model.alpha))

            sorted_topics = list(matutils.argsort(sort_alpha))
            chosen_topics = sorted_topics[:num_topics // 2] + sorted_topics[-num_topics // 2:]

        shown = []

        topic = model.state.get_lambda().shape
        print topic
        for i in chosen_topics:
            topic_ = topic[i]
            topic_ = topic_ / topic_.sum()  # normalize to probability distribution
            bestn = matutils.argsort(topic_, num_words, reverse=True)
            topic_ = [(model.id2word[id], topic_[id]) for id in bestn]
            if formatted:
                topic_ = ' + '.join(['%.3f*"%s"' % (v, k) for k, v in topic_])

            shown.append((i, topic_))
            if log:
                logger.info("topic #%i (%.3f): %s", i, model.alpha[i], topic_)

        return shown
    def perplexity(chunk):
        corpus = [self.vocab.doc2bow(pdoc) for pdoc in chunk]
        perplexity = model.model.log_perplexity(corpus)
        return perplexity



class SklearnBasedLda:
    """use module lda to implements LDA function
    """
    def train(self, vocab, doc_word_chunk, n_topics, n_iter = 2000):
        model = LatentDirichletAllocation(n_components = n_topics, max_iter = n_iter)
        self.n_topics = n_topics
        model.fit(doc_word_chunk[0])
        #more than one chunk
        if(len(doc_word_chunk) > 1):
            cnt = 1
            for doc_word in doc_word_chunk[1:]:
                logging.info('chunk: '+str(cnt))
                cnt += 1
                model.fit_transform(doc_word)
        self.model = model
        self.vocab = vocab

        #save to pickle
        timestamp = str(int(time.time()))
        save_path = os.path.join(model_path,'model_by_sklearn', 'iter_'+str(n_iter)+'_topics_'+str(n_topics)+'_'+timestamp)
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        pickle.dump(model, open(os.path.join(save_path,'model'),'w'))
        pickle.dump(vocab, open(os.path.join(save_path,'word_id_dict'),'w'))

    def load_model(self, path):
        self.model = pickle.load(file(os.path.join(path,'model')))
        self.dict = pickle.load(file(os.path.join(path,'word_id_dict')))

    def show_topics(self, n_top_words = 10):
        topic_word = self.model.components_
        #topic_word = self.model.topic_word_
        for i, topic_dist in enumerate(topic_word):
            topic_words = np.array(self.vocab)[np.argsort(topic_dist)][:-n_top_words - 1:-1]
            print('topic{}: {}'.format(i, ' '.join(topic_words)))

    def perplexity(chunk, chunk_size):
        dox_matrix = util.docs_to_matrix(chunk, self.word_id_dict, chunk_size)
        ret = self.model.perplexity(dox_matrix)
        return ret




if __name__ == '__main__':
    stop_words = utilities.read_stopwords('../data/aux/stop_words')

    vocab, doc_word = docs_to_matrix()
    #train()
    train_by_lda(vocab, doc_word)
    #predict()
    #model_file = os.path.join('../model/LDA', 'topic50_filter_again', 'LDA.model')
    #dict_file = '../data/aux/dict_no_below_20_no_above_05'
    #num_topics = 50
    #evaluate(model_file, dict_file, num_topics)
    #model = gensim.models.LdaModel.load(model_file)
    #show_topics(model)

    #query_bow = dictionary.doc2bow(jieba.cut(query, cut_all=False))
