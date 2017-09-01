# encoding=utf-8
import logging
import gensim
import tensorflow as tf
import sys
import time
import os
import cPickle
import jieba
reload(sys)
sys.path.append('..')
import utilities.utilities as utilities
sys.setdefaultencoding('utf-8')
logging.basicConfig(format = '%(asctime)s %(levelname)s : %(message)s', level = logging.INFO)
logging.getLogger().setLevel(logging.DEBUG)
#logging.root.level = logging.INFO

stop_words = {}
processed_docs = []

timestamp = str(int(time.time()))
flags = tf.app.flags
flags.DEFINE_string('save_path', os.path.join('../model/LDA/luru_fetched_news'), 'path for saving data')
FLAGS = flags.FLAGS


def predict():
    model = gensim.models.LdaModel.load('../model/LDA/luru_fetched_news/topic_50_LDA.model')
    model.print_topics(-1)
    dictionary = gensim.corpora.Dictionary.load('../model/LDA/luru_fetched_news/dict_no_below_20_no_above_1')

    query = u'马航代表与乘客家属见面'
    print(query)
    query_bow = dictionary.doc2bow(jieba.cut(query, cut_all=False))
    for index, score in sorted(model[query_bow], key = lambda tup : -1 * tup[1]):
        print "topic {} {} {}".format(index, score, model.print_topic(index, 10))

def train():
    topics = [50, 100, 200]
    stop_words = utilities.read_stopwords('../data/aux/stop_words')
    with open('../data/corpus/luru_fetched_news_processed.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            words = line.split(' ')
            words_list = []
            for word in words:
                if not stop_words.has_key(word):
                    words_list.append(word)
            processed_docs.append(words_list)
        word_cnt_dict = gensim.corpora.Dictionary(processed_docs)
        word_cnt_dict.filter_extremes(no_below = 20, no_above = 0.1)
        if not os.path.exists(FLAGS.save_path):
            os.makedirs(FLAGS.save_path)
        word_cnt_dict.save(os.path.join(FLAGS.save_path, 'dict_no_below_20_no_above_1'))

        bag_of_words_corpus = [word_cnt_dict.doc2bow(pdoc) for pdoc in processed_docs]
        logging.info('begin to trian model...')
        for topic in topics:
            lda_model = gensim.models.LdaModel(bag_of_words_corpus, num_topics=topic, id2word = word_cnt_dict)

            lda_model.save(os.path.join(FLAGS.save_path, 'topic_'+str(topic)+'_LDA.model'))
            print('topic num {} model finished'.format(topic))


if __name__ == '__main__':


    #with open('corpus_seg_content_all.txt', 'r') as f:
        # lines = f.readlines()
        # for line in lines:
        #     words = line.split(' ')
        #     words_list = []
        #     for word in words:
        #         if not stop_words.has_key(word):
        #             words_list.append(word)
        #     processed_docs.append(words_list)
        # word_cnt_dict = gensim.corpora.Dictionary(processed_docs)
        # #word_cnt_dict.filter_extremes(no_below = 40, no_above = 0.1)
        # for k,v in word_cnt_dict.items():
        #     print(k, v)
        # logging.info('witting word_cnt_dict')
        # cPickle.dump(word_cnt_dict, open('word_cnt_dict.p', 'wb'))
        # logging.info('finished!')
        #
        # bag_of_words_corpus = [word_cnt_dict.doc2bow(pdoc) for pdoc in processed_docs]
        # logging.info('bag_of_words_corpus')
        # cPickle.dump(bag_of_words_corpus, open('bag_of_words_corpus.p', 'wb'))
        # logging.info('finished')
        # logging.info('begin to trian model...')
        # lda_model = gensim.models.LdaModel(bag_of_words_corpus, num_topics=10, id2word = word_cnt_dict)
        # if not os.path.exists(FLAGS.save_path):
        #     os.makedirs(FLAGS.save_path)
        # lda_model.save(os.path.join(FLAGS.save_path, 'LDA.model'))
        # print('finished')
    # train()
    predict()
