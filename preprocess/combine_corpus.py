# encoding=utf-8
import os
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
path = '../data/corpus'
corpusPath = '/home/jane/NewsDatasets/Tencent_news_Result/'
files = os.listdir(corpusPath)


if __name__ == '__main__':
    with open(os.path.join(path, 'combine_corpus.txt'), 'a') as corpusAll:
        with open(os.path.join(path, 'pkbigdata_title_content_corpus.txt'),'r') as corpus1:
            corpusAll.write(corpus1.read())
        with open(os.path.join(path, 'sogou_title_content_corpus.txt'), 'r') as corpus2:
            corpusAll.write(corpus2.read())
