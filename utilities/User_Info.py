# encoding=utf-8
import logging
from copy import deepcopy
import numpy as np
from sets import Set
from Queue import PriorityQueue as PQueue

logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')

class UserInfo:
    """

    Attributes:
        userid:
        read_sum: total news number the user read
        topic_vec: a array of LDA probility distribution of the user has read in each topic
        recent_read: a series of tuple items saved in priority queue stands for the user
            recent read 10 pieces of news
            each tuple is like this: (timestamp, [, , ... , ]) means timestamp and title vector
            from word2vec of the recent read news
    """
    def __init__(self, user_id, recent_read_sum = 10):
        self.user_id = user_id
        self.read_sum = 0
        self.topic_vec = np.zeros(50)

        self.recent_read_sum = recent_read_sum
        self.recent_read = []
        self.read_list = Set()

    def update(self, news_info):
        self.update_recent_read(news_info)
        self.update_topic_vec(news_info.topic_dist)
        self.read_list.add(news_info.news_id)

    def update_recent_read(self, news_info):
        if len(self.recent_read) < self.recent_read_sum:
            self.recent_read.append(news_info)
        else:
            index = -1
            earist_timestamp = float('inf')
            for i, info in enumerate(self.recent_read):
                if info[0] < earist_timestamp:
                    earist_timestamp = info[0]
                    index = i
            if index != -1:
                self.recent_read[index] = news_info

    def update_topic_vec(self, topic_dist):
        self.topic_vec = (self.topic_vec * self.read_sum + topic_dist)/ (self.read_sum + 1)
        self.read_sum += 1

    def print_info(self):
        logging.info('----------------user info----------------------')
        logging.info('user id: {}'.format(self.user_id))
        logging.info('read sum: {}'.format(self.read_sum))
        logging.info('topic vec: {}'.format(self.topic_vec))
        logging.info('recent read: {}'.format(self.recent_read))
        logging.info('read list: {}'.format(self.read_list))
