# encoding=utf-8
import os
import csv
import sys
import re
import time
import jieba
import gensim
import copy
import User_Info
import News_Info
from copy import deepcopy
import numpy as np
from Queue import PriorityQueue as PQueue
from gensim import utils, matutils
from gensim.models import word2vec

def read_stopwords(stop_word_loc = '../data/aux/stop_words'):
    """read stop words from a stop words vocabulary

    Args:
        stop_word_loc: the stop words file location

    Returns:
        A dictionary including stop words
        example:

        {'之前'： 1，
         ‘，’： 1
        }
    """
    stop_words = {}

    with open(stop_word_loc,'r') as f:
        for line in f.readlines():
            stop_words[line.split('\n')[0]] = 1
    return stop_words

def set_field_size_limit():
    """set csv memory size limit, cause the default size is too small

    """
    maxInt = sys.maxsize
    decrement = True
    while decrement:
        # decrease the maxInt value by factor 10
        # as long as the OverflowError occurs.
        decrement = False
        try:
            csv.field_size_limit(maxInt)
        except OverflowError:
            maxInt = int(maxInt/10)
            decrement = True

def process_topic_vec(topic_vec):
    """process topic vector from string to float list

    Args:
        topic_vec: string

    Returns:
        a float list
    """
    topic_vec_string = re.split('\[|\]|,', topic_vec)
    topic_vec_string = topic_vec_string[1:len(topic_vec_string)-1]
    topic_vec_array = [float(value) for value in topic_vec_string]
    return topic_vec_array

def process_recent_read_queue(recent_read_queue):
    """process recent read info from string to priority queue of tuples
        (referred with User_Info recent_read_queue)

    Args:
        recent_read_queue: string

    Returns:
        a priority queue of tuples.
    """
    queue = PQueue()
    queue_string = recent_read_queue[1: len(recent_read_queue) - 1]
    queue_string_list = re.split('\(|\)', queue_string)
    queue_string_list = queue_string_list[1: len(queue_string_list) - 1]
    for value in queue_string_list:
        if(value == ', '):
            continue
        value.strip()
        value_list = re.split(',|\[|\]', value)
        timestamp = float(str(value_list[0]))
        title_list=[]
        for element in value_list[1:]:
            if(element == '' or element == ' '):
                continue
            title_list.append(float(element))
        queue.put((timestamp, title_list))
    return queue

def load_user_list(file):
    """load user info from user.csv and saved as UserInfo object
    """
    user_info_list = []
    user_id_dict = {}
    with open(file) as csvFile:
        rows = csv.DictReader(csvFile)
        for row in rows:
            user_id = str(row['userId'])
            read_sum = int(row['read_sum'])
            topic_vec = process_topic_vec(row['topic_vec'])
            recent_read_queue = process_recent_read_queue(row['recent_read_queue'])
            # print recent_read_queue
            user_info = User_Info.UserInfo(user_id, read_sum, np.array(topic_vec), recent_read_queue)
            index = len(user_id_dict)
            user_id_dict[user_id] = index
            user_info_list.append(user_info)
    return user_info_list, user_id_dict

def process_title(title, stop_words):
    # print title
    title_cut = jieba.cut(title, cut_all=False)
    title_filter = list(filter(lambda x: not stop_words.has_key(x.encode('utf-8')), title_cut))
    return title_filter

def load_news(file):
    """load sogou news in 2012.6 and saved as NewsInfo object
    """
    news_info_list = []
    with open(file, 'r') as txt:
        lines = txt.readlines()
        i = 0
        while i < len(lines) :
            pub_date_list = lines[i].split('/')
            if len(pub_date_list) < 3:
                i+=3
                continue
            pub_date = pub_date_list[-3]
            if not pub_date.isdigit() or len(pub_date) != 8:
                i+=3
                continue
            pub_data_formate = time.asctime()
            try:
                pub_data_formate = time.strptime(pub_date,'%Y%m%d')
            except:
                i+=3
                continue

            title = lines[i + 1].replace('<contenttitle>','')
            title = title.replace('</contenttitle>','')

            content = lines[i + 2].replace('<content>','')
            content = content.replace('</content>','')
            i += 3

            news_info = News_Info.NewsInfo(pub_data_formate, title, content)
            news_info_list.append(news_info)
    return news_info_list

def calc_topic_vec(news_list, LDA_model,corpus_dictionary, word2vec_model, stop_words):
    """calculate topic distribution and title vectors

        Args: news_list: a list of all NewsInfo type news
    """

    for index in range(len(news_list)):
        news_item = news_list[index]
        title_filter = process_title(news_item.title, stop_words)
        title_processed = corpus_dictionary.doc2bow(title_filter)
        topic_vec = [item[1] for item in LDA_model[title_processed]]
        news_list[index].topic_vec = topic_vec

        # calc word2vec
        word_vec = []
        for word in title_filter:
            if word in word2vec_model.vocab:
                word_vec.append(word2vec_model.wv[word])
        #word_vec =[word2vec_model.wv[word] for word in title_filter]
        title_vec = np.mean(word_vec, axis = 0).tolist()
        news_list[index].title_vec = title_vec

def find_user_topic_top_3(user):
    """for using LDA recommendation, first I find the top 3 topics the user most interest in,
    and then I choose some news relative to these 3 topics with a certain proportion

        Args:
            userInfo object

        Returns:
            a list of top 3 topic index

    """
    queue = PQueue()
    topic_vec = user.topic_vec.tolist()
    for i in range(len(topic_vec)):
        queue.put((-1 * topic_vec[i], i))
    return [queue.get()[1], queue.get()[1], queue.get()[1]]

def find_topic_top_news(today_news_list, topic, num, TOPIC_NUM):
    """find the most related news in given topic

        Args:
            today_news_list: news list
            topic: given topic index
            num: most related number of news you want to find

        Returns:
            a list of news id of most related ones
    """
    queue = PQueue()
    for i in range(len(today_news_list)):
        item = today_news_list[i]
        if(len(item.topic_vec) < TOPIC_NUM):
            continue
        queue.put((item.topic_vec[topic] * -1, i))

    top_list = []
    cnt = 0
    while not queue.empty() and cnt < num:
        item = queue.get()
        top_list.append(item[1])
        cnt += 1
    return top_list

def find_nearest_news(user, news_dict, num):
    """find nearest news using cosine distance

        Args:
            user: UserInfo object
            today_news_list: news list
            num: most related number of news you want to find

        Returns:
            a list of news id of most related ones

    """
    queue = PQueue()
    user_vec = user.recent_read
    user_vec = []
    for news_info in user.recent_read:
        user_vec.append(news_info.title_vec)

    for id, news_info in news_dict.items():
        title_vec = news_info.title_vec
        #cos_dist = np.dot(user_vec_mean, title_vec)
        cos_dist = np.dot(matutils.unitvec(np.array(user_vec).mean(axis=0)),
                   matutils.unitvec(np.array(title_vec)))

        queue.put((cos_dist * -1, id))
    top_list = []
    cnt = 0
    while not queue.empty() and cnt < num:
        item = queue.get()
        top_list.append(item[1])
        cnt += 1
    return top_list

def update_user_preference(user, news):
    read_sum = user.read_sum
    user_topic_vec = user.topic_vec
    recent_read_queue = user.recent_read_queue
    item_topic_vec_array = np.array(news.topic_vec)

    if(item_topic_vec_array.shape[0] == 10):
        user.user_topic_vec = user_topic_vec * read_sum / (read_sum + 1) + item_topic_vec_array * 1 / (read_sum + 1)
    user.read_sum = read_sum + 1
    if(recent_read_queue.qsize() >= 10):
        recent_read_queue.get()
    recent_read_queue.put((time.mktime(news.pub_time), news.title_vec))
    user.recent_read_queue = recent_read_queue
    return user
