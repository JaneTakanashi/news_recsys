# encoding = utf-8
import sys
reload(sys)
import logging
sys.setdefaultencoding('utf-8')
sys.path.append('..')
import pickle
import numpy as np
from recsys import recsys
from utilities.User_Info import UserInfo
from utilities.News_Info import NewsInfo

def show_rec_title(demo, rec_list):
    for id in rec_list:
        print demo.news_dict[id].title

if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level = logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    print logging.getLevelName(30)
    lda_model_file = '../model/LDA/topic50_filter_again_v2/LDA.model'
    dict_file = '../data/aux/dict_no_below_20_no_above_05_again_v2'
    word2vec_model_file = '../model/Word2Vec/1500975010/checkpoints/UScities.model.bin'
    demo = recsys(lda_model_file, dict_file, word2vec_model_file)

    # add user
    topic_num = 50
    user_pre = np.zeros(topic_num)

    # the number of save recent read news
    recent_read_sum = 10

    user_id = '321443'
    demo.user_entered([user_id], topic_num)

    rec_list = demo.get_recommend(user_id)
    show_rec_title(demo, rec_list)

    # click news
    demo.update_user_op('1002', 1503124610, '000f023ebc8e0fb51e22eb2679c269fd')
