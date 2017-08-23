# encoding=utf-8
import os
import sys
import time
import pickle
import logging
import threading
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

        stop_words_file = '../data/aux/stop_words'
        self.stop_words = utilities.read_stopwords(stop_words_file)
        self.user_info_file = '../database/user_list'
        self.news_info_file = '../database/news_list'

        lda_model = GensimBasedLda()
        lda_model.load_model(lda_model_file, dict_file)
        self.lda_model = lda_model

        logging.info('begin to load word2vec model...')
        self.word2vec_model = word2vec.KeyedVectors.load_word2vec_format(word2vec_model_file, binary=True)
        logging.info('word2vec model has been loaded successfully!')

        self.load_user_info()
        self.load_news_info()

        self.user_dict = {}
        self.news_dict = {}

        self.duration = duration
        self.update_hour = update_hour

        self.stopped = False
        self.t = threading.Thread(target = self.remove_news, name = 'remove news')
        self.t.start()

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
                    user_info = pickle.load(file(user_info_file))
                    user_id = user.user_id
                    self.user_dict[user_id] = user_info
                except:
                    has_pickle = False

    def load_news_info(self):
        with open(self.news_info_file, 'r') as pkfile:
            has_pickle = True
            while has_pickle:
                try:
                    news_info = pickle.load(file(user_info_file))
                    news_id = news_info.news_id
                    self.news_dict[user_id] = news_info
                except:
                    has_pickle = False

    def update_user_op(self, user_id, timestamp, operation):
        """
        user_id:
        op: operations like ('login', 'click', 'refresh')
        (ps: now only 'click' used)

        Operation Examples:
        ----------------------
             ['click', news_id)
             ['exit']
        """
        if operation[0] == 'click':
            if len(operation) < 2:
                logging.error('update user info with \'click\' but didn\'t provide news id')
                return
            news_id = operation[1]

            if not self.user_dict.has_key(user_id):
                logging.error('user key {} is not in news recsys database'.format(user_id))
                return
            if not self.news_dict.has_key(news_id):
                logging.error('news key {} is not in news recsys database'.format(news_id))
                return
            user_info = self.user_dict[user_id]
            news_info = self.news_dict[news_id]
            user_info.update(news_info)

    def add_user(self, user_info):
        """
            user_info: list
                contains user_id, preference
        """
        user_id = user_info[0]
        user_info = UserInfo(user_id)
        with open(self.news_info_file, 'a') as pkfile:
            pickle.dump(user_info, pkfile)
        self.user_dict[user_id] = user_info
        logging.info('add user {} successfully'.format(user_id))


    def add_news(self, news_info):
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
        rec_list = utilities.find_nearest_news(user_info, self.news_dict, num)
        return rec_list


    def remove_news(self):
        day_sec = 3600 * 24
        while not self.stopped:
            cur_time = time.time()
            cur_hour = time.localtime(time.time())[3]
            logging.info('remove info: it is {} o\'clock now, remove out-of-date news will happen at {} o\'clock'.format(cur_hour, self.update_hour))
            #remove out of date news at 2:00-3:00 am
            if cur_hour == self.update_hour:
                logging.info('begin to remove out of date news in recommend system...')
                for news_id, news_info in self.news_dict.items():
                    if int(cur_time) - news_info.pubtime > self.duration * day_sec:
                        with open(self.news_info_file, 'a') as pkfile:
                            pkfile.dump(news_dict[news_id], pkfile)
                        del self.news_dict[news_id]
                logging.info('remove finished! ')

            time.sleep(3600)
