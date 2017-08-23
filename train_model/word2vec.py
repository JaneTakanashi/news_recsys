# encoding=utf-8
import os
from gensim.models import word2vec
class Word2vec:
    def __init__(self):
        pass

    def load_model(self, model_file):
        if not os.path.isfile(model_file):
            logging.error('word2vector model file doesn\'t exits')
        return
        self.word2vec_model = word2vec.KeyedVectors.load_word2vec_format(model_path, binary=True)
