import os
import zipfile
import git

from data_providers.gikrya.gikrya_parser import read_conllu_file, get_pos_tags
from utils.data import is_done, download, mark_done


def build(datapath='downloads/gikrya'):
    if not is_done(datapath):
        print('Building pos')
        fname = 'GIKRYA_texts_new.zip'
        url = 'https://github.com/dialogue-evaluation/morphoRuEval-2017/raw/master/' + fname

        fname = os.path.join(datapath, fname)

        print('Downloading to {}'.format(fname))
        download(fname, url)

        print('Unzipping')
        with zipfile.ZipFile(fname, 'r') as zip:
            zip.extractall(path=datapath)

        os.remove(fname)

        mark_done(datapath)

        print('Built')
    return datapath


class DataProvider(object):
    def __init__(self, datapath='downloads/gikrya', lang='russian'):
        assert lang == 'russian'
        datapath = build(datapath)

        self.train_path = os.path.join(datapath, 'gikrya_new_train.out')
        self.train = list(read_conllu_file(self.train_path))
        self.train_pos_tags = list(get_pos_tags(self.train))
        self.dev_path = os.path.join(datapath, 'gikrya_new_test.out')
        self.dev = list(read_conllu_file(self.dev_path))
        self.dev_pos_tags = list(get_pos_tags(self.dev))

        self.pos_tags = sorted({tag for sent in self.train_pos_tags for word, tag in sent})


if __name__ == '__main__':
    build()
