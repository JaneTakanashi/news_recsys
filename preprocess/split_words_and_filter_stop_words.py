# encoding=utf-8
'''
preprocess news corpus
preprocess method:
    filterd html tag like <p></p>
    filterd stop words
    using jieba to cut words
@Time    : 17-8-30
@Author  : yangjie takanashiyj@gmail.com
'''

import csv
import jieba
import jieba.analyse
import re
import os
import timeit
import sys
reload(sys)
sys.path.append('..')
import utilities.utilities as utilities
sys.setdefaultencoding('utf-8')

path_raw = '../data/raw_data/'
path_processed = '../data/corpus/'
raw_file = 'luru_fetched_news.csv'

if __name__ == '__main__':
    utilities.set_field_size_limit()
    stop_words = utilities.read_stopwords('../data/aux/stop_words')
    content = []

    i = 0
    with open(os.path.join(path_raw, raw_file)) as csvFile:
        reader = csv.DictReader(csvFile)
        for row in reader:
            if i % 1000 == 0:
                print i
            i += 1
            content_re = re.sub('<.*?>|&nbsp|\n', '', row['content']).strip()
            title_re = re.sub('<.*?>|&nbsp|\n', '', row['title']).strip()
            content_re = content_re.strip('\n')
            title_re = title_re.strip('\n')
            seg_content = jieba.cut(content_re, cut_all = False)
            seq_title = jieba.cut(title_re, cut_all = False)

            for word in seq_title:
                if not stop_words.has_key(word.encode('utf-8')):
                    content.append(word)
                    content.append(' ')
            # content.append('\n')

            for word in seg_content:
                if not stop_words.has_key(word.encode('utf-8')):
                    content.append(word)
                    content.append(' ')
            content.append('\n')

    with open(os.path.join(path_processed, raw_file.split('.')[0]+'_processed.txt'),'w') as f:
        f.write(''.join(content))
    print('finished !')
