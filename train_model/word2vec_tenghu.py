# Author: Tenghu Wu
# Date:   2017-Jan-15

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

# set the global parameters here:
timestamp = str(int(time.time()))
flags = tf.app.flags
flags.DEFINE_string('data_path', './corpus_seg_all.txt', 'path for training data')
flags.DEFINE_string('save_path', os.path.join('./runs', timestamp, 'checkpoints'), 'path for saving data')
flags.DEFINE_integer('min_count', 2, 'term occurs less than this is ignored')
flags.DEFINE_integer('size', 50, 'embedding dimensions')
flags.DEFINE_integer('window', 4, 'terms occur within a window-neighborhood of a term')
flags.DEFINE_integer('sg', 1, 'sg=1:skip-gram model; sg=other:CBoW model')
# flags.DEFINE_float()
# flags.DEFINE_boolean()
FLAGS = flags.FLAGS

# the major part
if __name__ == '__main__':
    # logging information
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    # load-in training sentences
    sentences = word2vec.Text8Corpus(FLAGS.data_path)

    # training step:
    model = word2vec.Word2Vec(sentences,
                 	      min_count=FLAGS.min_count,
                              size=FLAGS.size,
                              window=FLAGS.window,
                              sg=FLAGS.sg)

    # save the trained model
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)
    model.save(os.path.join(FLAGS.save_path, 'UScities.model'))
    model.wv.save_word2vec_format(os.path.join(FLAGS.save_path, 'UScities.model.bin'), binary=True)
