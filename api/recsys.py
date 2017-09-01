# encoding=utf-8
import os
import re
import sys
import time
import math
import json
import pickle
import urllib
import logging
import threading
import numpy as np
from sets import Set
from gensim.models import word2vec
reload(sys)
sys.setdefaultencoding('utf-8')
sys.path.append('..')
import utilities.utilities as utilities
from utilities.User_Info import UserInfo
from utilities.News_Info import NewsInfo
from train_model.LDA import GensimBasedLda
"""
    news recommendation system api

    yangjie yangjie@interns.chuangxin.com
    2017.8.17
"""
class recsys:
    """

    Arrtibutes:
        user_dict: dict
            {userid: UserInfo}
        news_dict: dict
            {newsid: NewsInfo}

    """

    def __init__(self, lda_model_file, dict_file, word2vec_model_file, duration = 2, update_hour = 2):

        stop_words_file = './data/aux/stop_words'
        self.stop_words = utilities.read_stopwords(stop_words_file)
        self.user_info_file = './database/user_list'
        self.news_info_file = './database/news_list'
        self.web_log = './news/webapp/usedata/log.log'

        lda_model = GensimBasedLda()
        lda_model.load_model(lda_model_file, dict_file)
        self.lda_model = lda_model

        logging.info('begin to load word2vec model...')
        self.word2vec_model = word2vec.KeyedVectors.load_word2vec_format(word2vec_model_file, binary=True)
        logging.info('word2vec model has been loaded successfully!')

        self.user_dict = {}
        # load all news to calcuate user history preference
        self.news_dict = {}
        # recent few day news
        self.candidate_news_dict = {}
        # deprecated. get user info from analysing log now
        # self.load_user_info()
        self.load_news_info()

        #deprecated
        self.duration = duration
        self.update_hour = update_hour
        # self.stopped = False
        # self.t = threading.Thread(target = self.remove_news, name = 'remove news')
        # self.t.start()


    def delete(self):
        timestamp = int(time.time())
        with open(self.news_info_file, 'a') as pkfile:
            for news_info in self.news_dict:
                pickle.dump(news_info, pkfile)
        with open(self.user_info_file, 'w') as pkfile:
            for user_info in self.user_dict:
                pickle.dump(user_info, pkfile)
        self.stopped = True


    """load data from disk to memory

    """

    def load_word2vec_model(self, model_file):
        if not os.path.isfile(model_file):
            logging.error('word2vector model file doesn\'t exits')
        return
        word2vec_model = word2vec.KeyedVectors.load_word2vec_format(model_path, binary=True)
        return word2vec_model

    def load_user_info(self):
        with open(self.user_info_file, 'r') as pkfile:
            has_pickle = True
            while has_pickle:
                try:
                    user_info = pickle.load(pkfile)
                    user_id = user.user_id
                    self.user_dict[user_id] = user_info
                except:
                    has_pickle = False

    """ get all the news in database
        According to api document, I should get news and calc vector from the json given by url,
        However it is too slow, the cost of calculate vector is about 20s per 1000 pieces of news,
        which is not beneficial for debugging :(
        so I do the calculate process in advance and stored the results in pickle
    """
    def load_news_info(self):
        #in debug mode
        with open(self.news_info_file, 'r') as pkfile:
            has_pickle = True
            cur_time = int(time.time())
            day_sec = 3600 * 24
            while has_pickle:
                try:
                    news_info = pickle.load(pkfile)
                    news_id = news_info.news_id
                    pub_time = news_info.pub_time
                    self.news_dict[news_id] = news_info
                    if math.fabs(cur_time - int(pub_time)) < day_sec * 100:
                       self.candidate_news_dict[news_id] = news_info

                    # if math.abs(cur_time - self.news_dict[news_id].pub_time) > day_sec * 30:
                    #     print 'yes'
                    #     del self.news_dict[news_id].pub_time
                except:
                    has_pickle = False
        logging.info('load news from pickle finished, loaded {} pieces of news in all.'.format(len(self.news_dict)))

        # in running mode
        # cur_timestamp = int(time.time())
        # url = 'http://10.18.125.22:8000/webapp/api/time?start=0&end='+str(cur_timestamp)
        # filehandle = urllib.urlopen(url)
        # data = filehandle.read()
        # fetched_news = json.loads(data)
        # with open(self.news_info_file, 'a') as pkfile:
        #     for i, news in enumerate(fetched_news):
        #         if i % 1000 == 0:
        #             logging.info('add news {} successfully'.format(i))
        #         news_id = news['id']
        #         news_title = news['title']
        #         news_content = news['content']
        #         news_pub_time = news['pubDate']
        #         news_info = NewsInfo([news_id, news_title, news_content, news_pub_time], self.word2vec_model, self.lda_model, self.stop_words)
        #         #pickle.dump(news_info, pkfile)
        # logging.info('loaded news finished')

    """when user comes into our system
       firstly search in the log file to establish user history preferrence info
       then form an user record in memory
    """
    def user_entered(self, user_params, topic_num, recent_read_sum = 10):
        # step 1: form user info from web log
        print 'user entered!'
        user_id = user_params[0]
        # existing user
        if len(user_params) == 1:
            # this user info is already in memory
            if self.user_dict.has_key(user_id):
                user_info = self.user_dict[user_id]
            else:
                user_info = UserInfo(user_id, topic_num, np.zeros(topic_num))
                self.user_dict[user_id] = user_info

            with open(self.web_log, 'r') as log:
                lines = log.readlines()
                for line in lines:
                    element = line.split(',')
                    timestamp = int(element[0])
                    log_user_id = element[1]
                    if log_user_id == user_id:
                        # recommend info
                        if '[' in line:
                            # news_id is like u'9881481f3617dbfc4514dc0ea07dc13f', so I need to slice[2:-2]
                            recommend_news = map(lambda x: x.strip().strip('[').strip(']')[2:-1],element[2:])
                            for news_id in recommend_news:
                                user_info.recommend_list.add(unicode(news_id, 'utf-8'))
                        # user click info
                        elif len(element) == 3:
                            news_id = unicode(re.sub('\n','',element[2]), 'utf-8')
                            if not self.news_dict.has_key(news_id):
                                logging.error('import log error: news_id {} doesn\'t exists in news json'.format(news_id))
                            else:
                                user_info.update(self.news_dict[news_id])
        # new users
        elif len(user_params) == 2:
            print 'new user entered!'
            user_info = UserInfo(user_id, topic_num, user_params[1])
            self.user_dict[user_id] = user_info

        # step 2: get candidate news list
        self.user_dict[user_id].fetch_candidate_news()



    def update_user_op(self, user_id, timestamp, news_id):
        """
        user_id:
        op: operations like ('login', 'click', 'refresh')
        (ps: now only 'click' used)

        Operation Examples:
        ----------------------
             'click': news_id
             'exit'
        """
        if not self.user_dict.has_key(user_id):
            logging.error('user key {} is not in news recsys database'.format(user_id))
            return
        if not self.news_dict.has_key(news_id):
            logging.error('news key {} is not in news recsys database'.format(news_id))
            return
        user_info = self.user_dict[user_id]
        news_info = self.news_dict[news_id]
        user_info.update(news_info)

    # deprecated
    def add_user(self, user_id, topic_num, recent_read_sum = 10):
        """
            user_info: list
                contains user_id, preference
        """
        user_info = UserInfo(user_id, topic_num, recent_read_sum)
        with open(self.news_info_file, 'a') as pkfile:
            pickle.dump(user_info, pkfile)
        self.user_dict[user_id] = user_info
        logging.info('add user {} successfully'.format(user_id))


    def add_news(self, user_idnews_info):
        """
        news_info: [news_id, title, content, timestamp]

        """
        news_instance = NewsInfo(news_info, self.word2vec_model, self.lda_model, self.stop_words)
        news_id = news_info[0]
        self.news_dict[news_id] = news_instance
        logging.info('add news {} successfully'.format(news_id))

    def search_user(self, user_id):
        if self.user_dict.has_key(user_id):
            self.user_dict[user_id].print_info()
        else:
            logging.info('user {} doesn\'t exists'.format(user_id))

    def search_news(self, news_id):
        if self.news_dict.has_key(news_id):
            self.news_dict[news_id].print_info()
        else:
            logging.info('news {} doesn\'t exists'.format(news_id))

    def get_recommend(self, user_id, num = 10):
        rec_list = []
        user_info = self.user_dict[user_id]
        print 'candidate news num: '+ str(len(self.candidate_news_dict))
        print user_info.topic_vec
        rec_list_by_word2vec = utilities.find_nearest_news(user_info, self.candidate_news_dict, num)
        rec_list_by_lda = utilities.find_topic_top_news(user_info, self.candidate_news_dict, num)
        rec_list = Set(rec_list_by_word2vec + rec_list_by_lda)
        self.user_dict[user_id].recommend_list |= rec_list
        return list(rec_list)

    # deprecate
    # def remove_news(self):
    #     day_sec = 3600 * 24
    #     while not self.stopped:
    #         cur_time = time.time()
    #         cur_hour = time.localtime(time.time())[3]
    #         logging.info('remove info: it is {} o\'clock now, remove out-of-date news will happen at {} o\'clock'.format(cur_hour, self.update_hour))
    #         #remove out of date news at 2:00-3:00 am
    #         if cur_hour == self.update_hour:
    #             logging.info('begin to remove out of date news in recommend system...')
    #             for news_id, news_info in self.news_dict.items():
    #                 if int(cur_time) - news_info.pubtime > self.duration * day_sec:
    #                     with open(self.news_info_file, 'a') as pkfile:
    #                         pkfile.dump(news_dict[news_id], pkfile)
    #                     del self.news_dict[news_id]
    #             logging.info('remove finished! ')
    #
    #         time.sleep(3600)
