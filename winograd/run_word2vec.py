PACKAGE_PATH = './packages'
PARSER_PATH = '/project/comp5211'
WORD2VEC_PATH = './packages/word2vec/lib/python2.7/site-packages'
GENSIM_PATH = './packages/gensim/lib/python2.7/site-packages'

import sys
if not PACKAGE_PATH in sys.path:
    sys.path.append(PACKAGE_PATH)
    sys.path.append(PARSER_PATH)
    sys.path.append(WORD2VEC_PATH)
    sys.path.append(GENSIM_PATH)

import word2vec
import gensim

model = gensim.models.Word2Vec.load_word2vec_format('/project/svm/tf_lstm/GoogleNews-vectors-negative300.bin', binary=True)
print(model.similarity('woman', 'man'))
