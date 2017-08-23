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


if __name__ == '__main__':

    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', level = logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)
    print logging.getLevelName(30)
    lda_model_file = '../model/LDA/topic50_filter_again_v2/LDA.model'
    dict_file = '../data/aux/dict_no_below_20_no_above_05_again_v2'
    word2vec_model_file = '../model/Word2Vec/1500975010/checkpoints/UScities.model.bin'
    demo = recsys(lda_model_file, dict_file, word2vec_model_file)

    # add user
    print 'add user-------------------------------'
    user_pre = np.zeros(50)
    demo.add_user(['1002', user_pre])
    demo.search_user('1002')

    # add news
    for i in range(50):
        news = [str(i), 'the 91th birthday year for Mr.Jiang','It is Mr.Jiang\'s ......Aug 17, 2017',23333333]
        demo.add_news(news)
    demo.search_news('1')


    # click news
    demo.update_user_op('1002', 1503124610, ['click', '1'])
    demo.search_user('1002')

    rec_list = demo.get_recommend('1002')
    print rec_list

    demo.delete()
