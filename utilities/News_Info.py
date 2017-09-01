import utilities
import logging
import numpy as np
logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s')

class NewsInfo:
    """
    parameters:
        news_info is a list contains pubtime, title and content
        Examples:
        -------------------------------------------------------
        ['1000', 'the 91th birthday year for Mr.Jiang',
        'It is Mr.Jiang's .............. Aug 17, 2017',23333333]

    Attributes:
        pub_time: publish time
        title_vec: title vector using word2vec
        topic vec: a array of LDA probility distribution of news topic
    """
    def __init__(self, news_info, word2vec_model, lda_model, stop_words):
        if len(news_info) < 4:
            raise Exception('Invalid news_info')

        self.news_id = news_info[0]
        self.title = news_info[1]
        self.content = news_info[2]
        self.pub_time = int(news_info[3])

        self.calc_title_vec(word2vec_model, stop_words)
        self.calc_topic_dist(lda_model, stop_words)

    def calc_title_vec(self, word2vec_model, stop_words):
        title_list = utilities.process_title(self.title, stop_words)
        # add content
        content_list = utilities.process_title(self.content, stop_words)
        title_list += content_list

        word_vec = []
        for word in title_list:
            if word in word2vec_model.vocab:
                word_vec.append(word2vec_model.wv[word])
        #word_vec =[word2vec_model.wv[word] for word in title_filter]
        self.title_vec = np.mean(word_vec, axis = 0).tolist()

    def calc_topic_dist(self, lda_model, stop_words):
        title_list = utilities.process_title(self.title, stop_words)
        content_list = utilities.process_title(self.content, stop_words)
        self.topic_dist = lda_model.get_topic_distribution(title_list + content_list)

    def print_info(self):
        logging.info('----------------news info----------------------')
        logging.info('news id: {}'.format(self.news_id))
        logging.info('title: {}'.format(self.title))
        logging.info('content: {}'.format(self.content))
        logging.info('pub_time: {}'.format(self.pub_time))
        logging.info('title_vec: {}'.format(self.title_vec))
        logging.info('topic_dist: {}'.format(self.topic_dist))
