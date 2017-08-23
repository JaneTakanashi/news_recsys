# encoding=utf-8
# Author: Tenghu Wu
# Date: 2017-Jan-15
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from gensim.models import word2vec
from wikipedia import search, page

import logging
import numpy as np
import os
import tensorflow as tf
import time
#import sys
#reload(sys)
#sys.setdefaultencoding('utf-8')

# set the global parameters here:
timestamp = str(int(time.time()))
flags = tf.app.flags
flags.DEFINE_string('data_path', './corpusAll.txt', 'path for training data')
flags.DEFINE_string('save_path', 'runs/1500969524/checkpoints', 'path for saving data')
flags.DEFINE_integer('min_count', 2, 'term occurs less than this is ignored')
flags.DEFINE_integer('size', 50, 'embedding dimensions')
flags.DEFINE_integer('window', 4, 'terms occur within a window-neighborhood of a term')
flags.DEFINE_integer('sg', 1, 'sg=1:skip-gram model; sg=other:CBoW model')
# flags.DEFINE_float()
# flags.DEFINE_boolean()
FLAGS = flags.FLAGS

# the major part
if __name__ == '__main__':
    # load-in trained model
    file_path = os.path.join(FLAGS.save_path,'UScities.model.bin')

    #model = word2vec.Word2Vec.load_word2vec_format(file_path, binary=True)
    model = word2vec.KeyedVectors.load_word2vec_format(file_path, binary=True)
    word = u'李开复'
    print('similiar with '+word)
    # test example 1:
    for sim in model.most_similar(word):
    	print(sim[0].encode('utf-8'), sim[1])
    print(model[word])
