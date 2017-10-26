import os
from collections import namedtuple

ONTONOTES_FOLDER = os.path.expanduser('downloads/OntoNotes')
LANG_FOLDER = os.path.join(ONTONOTES_FOLDER, 'data', 'files', 'data')
LANGS = {'arabic', 'chinese', 'english'}

CONLL_TEST_FILES = 'downloads/conll-2012-test.txt'


def _get_test_files_list():
    conll_test = set()
    with open(CONLL_TEST_FILES) as f:
        for line in f:
            conll_test.add(line.strip())
    return conll_test

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


Conll = namedtuple('Conll', 'id form head')


import os
import subprocess
import sys
import shutil
import tarfile
import git

from utils.data import download


def _convert(py_cleannlp_path, ontonotes_folder):
    sys.path.append(py_cleannlp_path)

    from clearnlp.converter import SubprocessConverter

    c = SubprocessConverter()

    command = [c.java_command,
               '-ea',
               '-cp', ':'.join(c.classpath),
               c.class_name,
               '-h', c.head_rule_path,
               '-r',
               '-i', ontonotes_folder]
    proc = subprocess.run(command,
                          stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE,
                          universal_newlines=True)

    return proc.returncode


def prepare_ontonotes_dataset(onto_folder):
    ontonotes_archive = 'downloads/OntoNotes.tar.gz'

    print('Downloading OntoNotes from Nexus storage...', file=sys.stderr)
    download(ontonotes_archive, 'http://share.ipavlov.mipt.ru:8080/repository/datasets/OntoNotes.tar.gz')
    assert os.path.isfile(ontonotes_archive)

    print('Unpacking received file...', file=sys.stderr)
    with tarfile.open(ontonotes_archive) as f:
        f.extractall('downloads')
    assert os.path.isdir(onto_folder)

    os.remove(ontonotes_archive)
    assert not os.path.isfile(ontonotes_archive)

    print('Deleting all files but with "parse" extension...', file=sys.stderr)
    for root, dirs, files in os.walk(onto_folder):
        for fn in files:
            if not fn.endswith('.parse'):
                os.remove(os.path.join(root, fn))

    clearnlp_converter = 'downloads/py-clearnlp-converter'
    if not os.path.isdir(clearnlp_converter):
        print('Downloading ''py-clearnlp-converter'' from git', file=sys.stderr)
        git.Git().clone('https://github.com/honnibal/py-clearnlp-converter')
        shutil.move('py-clearnlp-converter', clearnlp_converter)

    print('Converting constituents into dependencies...', file=sys.stderr)
    _convert(clearnlp_converter, onto_folder)

    print('Downloading a list of test files from UD 2.0...', file=sys.stderr)
    download('downloads/conll-2012-test.txt',
             'http://share.ipavlov.mipt.ru:8080/repository/datasets/conll-2012-test.txt')

    print('Done!', file=sys.stderr)


def _read_file(fn):
    with open(fn) as f:
        words = []
        for line in f:
            line = line.strip()
            if line:
                row = line.split('\t')
                idx = row[0]
                form = row[1]
                head = row[5]
                w = Conll(idx, form, head)
                words.append(w)
            else:
                yield words
                words = []
    if words:
        yield words


def _read_data(lang):
    conll_test = _get_test_files_list()

    train = []
    test = []
    for root, dirs, files in os.walk(os.path.join(LANG_FOLDER, lang)):
        for fn in files:
            if not fn.endswith('.parse.dep'):
                continue
            if fn[:-len('.parse.dep')] in conll_test:
                target = test
            else:
                target = train

            target.extend(_read_file(os.path.join(root, fn)))
    return train, test


class DataProvider(object):
    def __init__(self, lang='english'):
        assert lang in {'english', 'arabic', 'chinese'}
        if not os.path.isdir(ONTONOTES_FOLDER):
            prepare_ontonotes_dataset(ONTONOTES_FOLDER)

        self.train, self.dev = _read_data(lang)
