# encoding=utf-8
import time
import json
import urllib
import logging
from copy import deepcopy
import numpy as np
from sets import Set
from News_Info import NewsInfo

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
    def __init__(self, user_id, topic_num, topic_vec, recent_read_sum = 10):
        self.user_id = user_id
        self.read_sum = 1
        self.topic_vec = topic_vec

        self.recent_read_sum = recent_read_sum
        self.recent_read = []
        # all news id this user has read
        self.read_list = Set()
        # news has recommended to this user
        self.recommend_list = Set()
        # fetched candidate news
        self.candidate_list = Set()

    """ fetch candidate news from url
        fetched news is in the interval[last_read_timestamp, now]
        all the candidate news id are stored in self.cadidate_list

    """
    def fetch_candidate_news(self):
        cur_timestamp = int(time.time())
        _, recent_latest_timestamp = self.get_latest()
        # if the user hasn't read any user, recommend recent 3 days news
        if recent_latest_timestamp == -1:
            recent_latest_timestamp = cur_timestamp - 3600 * 24 * 3

        # recent_latest_timestamp = 1502541259
        print 'candidate news interval: '+str(recent_latest_timestamp)+' '+str(cur_timestamp)

        # bug, user other time interval can't fetch news
        recent_latest_timestamp = 0
        url = 'http://10.18.125.40:8001/webapp/api/time?start='+str(recent_latest_timestamp)+'&end='+str(cur_timestamp)
        print url
        filehandle = urllib.urlopen(url)
        data = filehandle.read()
        fetched_news = json.loads(data)
        print len(fetched_news)
        for news in fetched_news:
            self.candidate_list.add(news['id'])
        logging.info('candidate news num: {}'.format(len(self.candidate_list)))

    """return latest timestamp the user read news
    """
    def get_latest(self):
        # the user didn't read any news
        if not len(self.recent_read):
            return -1, -1
        latest_timestamp = self.recent_read[0].pub_time
        index = 0
        for i, info in enumerate(self.recent_read):
            if info.pub_time > latest_timestamp:
                earist_timestamp = info.pub_time
                index = i
        return index, self.recent_read[index].pub_time

    def get_earlist(self):
        # the user didn't read any news
        if not len(self.recent_read):
            return -1, -1
        index = -1
        earist_timestamp = float('inf')
        for i, info in enumerate(self.recent_read):
            if int(info.pub_time) < earist_timestamp:
                earist_timestamp = int(info.pub_time)
                index = i
        return index, self.recent_read[index].pub_time

    def update(self, news_info):
        self.update_recent_read(news_info)
        self.update_topic_vec(news_info.topic_dist)
        self.read_list.add(news_info.news_id)

    def update_recent_read(self, news_info):
        if len(self.recent_read) < self.recent_read_sum:
            self.recent_read.append(news_info)
        else:
            index, _ = self.get_earlist()
            self.recent_read[index] = news_info

    def update_topic_vec(self, topic_dist):
        self.topic_vec = (self.topic_vec * self.read_sum + topic_dist)/ (self.read_sum + 1)
        print topic_dist
        print self.topic_vec
        self.read_sum += 1

    def print_info(self):
        logging.info('----------------user info----------------------')
        logging.info('user id: {}'.format(self.user_id))
        logging.info('read sum: {}'.format(self.read_sum))
        logging.info('topic vec: {}'.format(self.topic_vec))
        logging.info('recent read: {}'.format(self.recent_read))
        logging.info('read list: {}'.format(self.read_list))
