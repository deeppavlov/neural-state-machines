import glob
import os
import tarfile

from utils.data import download, is_done, mark_done
from .conllu_parser import read_conllu_file, get_pos_tags


def build(datapath='downloads'):
    result_path = os.path.join(datapath, 'ud-treebanks-v2.0')

    if not is_done(result_path):
        print('Building pos')
        fname = 'ud-treebanks-v2.0.tgz'
        url = 'https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-1983/' + fname

        fname = os.path.join(datapath, fname)

        print('Downloading to {}'.format(fname))
        download(fname, url)

        print('Untarring')
        with tarfile.open(fname) as tar:
            tar.extractall(path=datapath)

        os.remove(fname)

        mark_done(result_path)

        print('Built')
    return result_path


languages = {
    'english': 'UD_English',
    'russian': 'UD_Russian'
}


class DataProvider(object):
    def __init__(self, datapath='downloads', lang='english'):
        datapath = build(datapath)

        if lang not in languages:
            assert RuntimeError('Not a supported language')
        lang = languages[lang]

        root = os.path.join(datapath, lang)

        self.train_path = glob.glob(os.path.join(root, '*train.conllu'))[0]
        self.train = list(read_conllu_file(self.train_path))
        self.train_pos_tags = list(get_pos_tags(self.train))
        self.dev_path = glob.glob(os.path.join(root, '*dev.conllu'))[0]
        self.dev = list(read_conllu_file(self.dev_path))
        self.dev_pos_tags = list(get_pos_tags(self.dev))

        self.pos_tags = sorted({tag for sent in self.train_pos_tags for word, tag in sent})


if __name__ == '__main__':
    build()
