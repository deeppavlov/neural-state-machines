import os
import sys
from collections import namedtuple

CONVERTED_ONTONOTES_FOLDER = os.path.expanduser('~/data/OntoNotesConll')
LANG_FOLDER = os.path.join(CONVERTED_ONTONOTES_FOLDER, 'data', 'files', 'data')
LANGS = {'arabic', 'chinese', 'english'}

CONLL_FILE_LIST_DIR = 'data_providers/OntoNotes'
CONLL_TEST_FILES = os.path.join(CONLL_FILE_LIST_DIR, 'conll-2012-test.txt')
CONLL_TRAIN_FILES = os.path.join(CONLL_FILE_LIST_DIR, 'conll-2012-train.txt')


conll_test = set()
with open(CONLL_TEST_FILES) as f:
    for line in f:
        conll_test.add(line.strip())

conll_train = set()
with open(CONLL_TRAIN_FILES) as f:
    for line in f:
        conll_train.add(line.strip())

# conll_test = os.path.expanduser('~/data/conll-2012/test')
# test_files = set()
# for root, dirs, files in os.walk(conll_test):
#     for fn in files:
#         filename, extension = fn.rsplit('.', 1)
#         if extension == 'v4_gold_conll':
#             test_files.add(filename)
# with open('conll-2012-test.txt', 'w') as f:
#     for t in test_files:
#         print(t, file=f)
#
# conll_train = os.path.expanduser('~/data/conll-2012/train')#
# train_files = set()
# for root, dirs, files in os.walk(conll_test):
#     for fn in files:
#         filename, extension = fn.rsplit('.', 1)
#         if extension == 'v4_gold_conll':
#             train_files.add(filename)
# with open('conll-2012-train.txt', 'w') as f:
#     for t in train_files:
#         print(t, file=f)

Conll = namedtuple('Conll', 'id form lemma cpostag unk head deplabel')


def _read_file(fn):
    with open(fn) as f:
        words = []
        for line in f:
            line = line.strip()
            if line:
                w = Conll(*line.split('\t')[:-3])
                words.append(w)
            else:
                yield words
                words = []
    if words:
        yield words


def _read_data(lang):
    train = []
    test = []
    for root, dirs, files in os.walk(os.path.join(LANG_FOLDER, lang)):
        for fn in files:
            if not fn.endswith('parse.dep'):
                continue
            if fn in conll_test:
                target = test
            else:
                target = train

            target.extend(_read_file(os.path.join(root, fn)))
    return train, test


class DataProvider(object):
    def __init__(self, lang='english'):
        assert lang in {'english', 'arabic', 'chinese'}
        self.train, self.dev = _read_data(lang)
