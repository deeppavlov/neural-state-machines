import random
from collections import defaultdict, Iterator
from itertools import chain

import os

import pickle

from data_providers.ud_pos import pos as ud
import data_providers.ontonotes as onto

WORD_ROOT = ' '
WORD_EMPTY = '\t'
WORD_EMPTY_ID = 2
WORD_ROOT_ID = 1
WORD_UNKNOWN_ID = 0

BUCKETS_COUNT = 8


def batch_generator(seq: Iterator, batch_size):
    seq = list(seq)
    while True:
        # indexi = torch.randperm(len(seq))
        # for i in range(0, len(indexi), batch_size):
        #     yield [seq[k] for k in indexi[i:i+batch_size]]

        random.shuffle(seq)
        bucket_size = 1 + (len(seq) - 1) // BUCKETS_COUNT
        buckets = [seq[i * bucket_size:(1 + i) * bucket_size] for i in range(BUCKETS_COUNT)]

        buckets = [ex for i, bucket in enumerate(buckets) for ex in sorted(bucket, reverse=bool(i % 2))]

        for i in range(0, len(buckets), batch_size):
            yield buckets[i:i + batch_size]


def create_dictionary(words, reserved_ids=None, min_count=2):
    if reserved_ids is None:
        reserved_ids = {}
    word_counts = defaultdict(int)
    word_ids = dict()
    next_id = max(chain([-1], reserved_ids.values())) + 1

    for w in words:
        word_counts[w] += 1
        if w not in word_ids:
            word_ids[w] = next_id
            next_id += 1

    # shrink resulting dictionary
    result = dict(reserved_ids)
    next_id = max(chain([-1], reserved_ids.values())) + 1
    for w, i in word_ids.items():
        if word_counts[w] >= min_count:
            result[w] = next_id
            next_id += 1

    return result


def cached(cache_filename, creating_function, rewrite=False):
    if not os.path.isfile(cache_filename) or rewrite:
        data = creating_function()
        print('Creating cache file "{}"'.format(cache_filename))
        with open(cache_filename, 'wb') as f:
            pickle.dump(data, f)

    print('loading data from "{}"'.format(cache_filename))
    with open(cache_filename, 'rb') as f:
        return pickle.load(f)


def prepare_data(provider='onto', lang='english'):
    providers = {
        'onto': onto,
        'ud': ud
    }
    if provider not in providers:
        raise RuntimeError('unknown data provider')

    conllu = providers[provider].DataProvider(lang=lang)

    dictionary = create_dictionary(chain(*([w.form for w in s] for s in conllu.train)),
                                   reserved_ids={'_UKNOWN_': WORD_UNKNOWN_ID, WORD_ROOT: WORD_ROOT_ID,
                                                 WORD_EMPTY: WORD_EMPTY_ID})
    print('Dictionary has {} elements'.format(len(dictionary)))

    tags_dictionary = create_dictionary(chain(*([w.postag for w in s] for s in conllu.train)),
                                        reserved_ids={'_UKNOWN_': WORD_UNKNOWN_ID, WORD_ROOT: WORD_ROOT_ID,
                                                      WORD_EMPTY: WORD_EMPTY_ID},
                                        min_count=1)
    print('Tags dictionary has {} elements'.format(len(tags_dictionary)))

    deprel_dictionary = create_dictionary(chain(*([w.deprel for w in s] for s in conllu.train)),
                                          reserved_ids={'_UKNOWN_': WORD_UNKNOWN_ID},
                                          min_count=1)
    print('Head labels dictionary has {} elements'.format(len(deprel_dictionary)))

    train = []
    train_gold_errors = 0
    for s in conllu.train:
        try:
            sent = []
            for w in s:
                int(w.id)  # crash if not integer
                sent.append((dictionary.get(w.form, WORD_UNKNOWN_ID), w.form,
                             tags_dictionary[w.postag], w.postag,
                             int(w.head), deprel_dictionary[w.deprel]))
            train.append(sent)
        except ValueError:
            train_gold_errors += 1
    print('Train has {} examples after removing {} decimal errors'.format(len(train), train_gold_errors))

    test = []
    test_gold_errors = 0
    for s in conllu.dev:
        try:
            sent = []
            for w in s:
                int(w.id)
                sent.append((dictionary.get(w.form, WORD_UNKNOWN_ID), w.form,
                             tags_dictionary.get(w.postag, WORD_UNKNOWN_ID), w.postag,
                             int(w.head), deprel_dictionary[w.deprel]))
            test.append(sent)
        except ValueError:
            test_gold_errors += 1
    print('Test has {} examples after removing {} decimal errors'.format(len(test), test_gold_errors))

    return train, test, dictionary, tags_dictionary, deprel_dictionary