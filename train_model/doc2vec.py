# encoding=utf-8
import gensim
import os
import collections
import smart_open
import random
import logging
import tensorflow as tf
import time
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

timestamp = str(int(time.time()))
flags = tf.app.flags
flags.DEFINE_string('save_path', os.path.join('./runs/Doc2Vec', timestamp, 'checkpoints'), 'path for saving data')
FLAGS = flags.FLAGS

def read_corpus(fname, tokens_only=False):
    with smart_open.smart_open(fname, encoding="iso-8859-1") as f:
        for i, line in enumerate(f):
            if tokens_only:
                yield gensim.utils.simple_preprocess(line)
            else:
                # For training data, add tags
                yield gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(line), [i])

if __name__ == '__main__':
    train_corpus = list(read_corpus('content-small'))
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.doc2vec.Doc2Vec(size=50, min_count=2, iter=55)
    model.build_vocab(train_corpus)

    # save the trained model
    if not os.path.exists(FLAGS.save_path):
        os.makedirs(FLAGS.save_path)
    model.save(os.path.join(FLAGS.save_path, 'UScities.model'))
