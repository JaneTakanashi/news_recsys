# encoding=utf-8
"""更新停词表后重新过滤一遍
"""
import sys
import os
import re
reload(sys)
sys.path.append('..')
import utilities.utilities as utilities
sys.setdefaultencoding('utf-8')

file_source = '../data/corpus/combine_corpus.txt'
file_des = '../data/corpus/combine_corpus_filter_again.txt'
stop_words_file = '../data/aux/stop_words'
if __name__ == '__main__':
    stop_words = utilities.read_stopwords('../data/aux/stop_words')
    i = 0
    with open(file_des, 'w') as f_des:
        with open(file_source, 'r') as f_sou:
            lines = f_sou.readlines()
            for line in lines:
                i += 1
                if i % 1000 == 0:
                    print i
                words = re.split(' |\n', line)
                word_filter = []
                for word in words:
                    if not stop_words.has_key(word.encode('utf-8')):
                        word_filter.append(word)
                word_filter.append('\n')
                f_des.write(' '.join(word_filter))
