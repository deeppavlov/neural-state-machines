import os
import tarfile

from utils.data import download, is_done, mark_done


def build(datapath='downloads'):
    result_path = os.path.join(datapath, 'ud-treebanks-v2.0')

    if is_done(result_path):
        return

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


if __name__ == '__main__':
    build()
