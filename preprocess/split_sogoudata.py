# -*- coding: utf-8 -*-
# word_segment.py用于语料分词

import logging
import os.path
import sys
import re
import jieba
reload(sys)
sys.path.append('..')
import utilities.utilities as utilities
sys.setdefaultencoding( "utf-8" )

# 先用正则将<content>和</content>去掉
def reTest(content):
  reContent = re.sub('<content>|</content>|<contenttitle>|</contenttitle>','',content)
  return reContent

title_uniq_dict = {}
if __name__ == '__main__':
  program = os.path.basename(sys.argv[0])
  logger = logging.getLogger(program)
  logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
  logging.root.setLevel(level=logging.INFO)
  logger.info("running %s" % ' '.join(sys.argv))
  stop_words = utilities.read_stopwords('../data/aux/stop_words')

   # check and process input arguments
  if len(sys.argv) < 3:
    print globals()['__doc__'] % locals()
    sys.exit(1)
  inp, outp = sys.argv[1:3]
  space = " "
  i = 0

  finput = open(inp)
  foutput = open(outp,'w')
  lines = finput.readlines()
  uniq = 0
  for i in range(len(lines)):
    line = lines[i]
    if(i % 2 == 0):
        #title line
        if(title_uniq_dict.has_key(line)):
            uniq = 0
            continue
        else:
            title_uniq_dict[line] = 1
            uniq = 1
    if(uniq == 1):

        line_seg = jieba.cut(reTest(line))
        filter_words = []

        for words in line_seg:
            if not stop_words.has_key(words.encode('utf-8')):
                filter_words.append(words)
        foutput.write(space.join(filter_words))
        if (i % 1000 == 0):
          logger.info("Saved " + str(i) + " articles_seg")

  finput.close()
  foutput.close()
  logger.info("Finished Saved " + str(i) + " articles")
