import os
import random
import pickle
from collections import namedtuple, defaultdict
from copy import deepcopy
from itertools import chain
from time import time
from typing import List, Iterator, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F

from data_providers.ud_pos import pos as ud
import data_providers.ontonotes as onto

STOP_AFTER_SAMPLES = 20000 * 1000
TEST_EVERY_SAMPLES = 5000
CHAR_EMB_COUNT = 5000
TAG_EMB_COUNT = 100
HIDDEN_SIZE = 200

WORD_DIM = 20
CHAR_DIM = 20
LABEL_DIM = 12
TAG_DIM = 20
OUT_DIM = 3
INPUT_DIM = (WORD_DIM + CHAR_DIM + TAG_DIM) * (3 + 3)
# INPUT_DIM = (WORD_DIM + CHAR_DIM + TAG_DIM) * (3 + 3 + 4)

# TOP_FROM_STACK = 3
# TOP_FROM_BUFFER = 3

LR = 0.005
BATCH_SIZE = 64
L2_DECAY = 1e-6

WORD_UNKNOWN = '_UKNOWN_'
WORD_ROOT = ' '
WORD_EMPTY = '\t'

WORD_UNKNOWN_ID = 0
WORD_ROOT_ID = 1
WORD_EMPTY_ID = 2

BUCKETS_COUNT = 8

PUNISH = 10
SHIFT_BASIC_ERROR = 3


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


assert create_dictionary('a b c a c c c'.split(), reserved_ids={' ': 0}, min_count=2) in [{' ': 0, 'a': 1, 'c': 2},
                                                                                          {' ': 0, 'a': 2, 'c': 1}]


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
                                   reserved_ids={WORD_UNKNOWN: WORD_UNKNOWN_ID, WORD_ROOT: WORD_ROOT_ID,
                                                 WORD_EMPTY: WORD_EMPTY_ID})
    print('Dictionary has {} elements'.format(len(dictionary)))

    tags_dictionary = create_dictionary(chain(*([w.postag for w in s] for s in conllu.train)),
                                        reserved_ids={WORD_UNKNOWN: WORD_UNKNOWN_ID, WORD_ROOT: WORD_ROOT_ID,
                                                      WORD_EMPTY: WORD_EMPTY_ID},
                                        min_count=1)
    print('Tags dictionary has {} elements'.format(len(tags_dictionary)))

    deprel_dictionary = create_dictionary(chain(*([w.deprel for w in s] for s in conllu.train)),
                                          reserved_ids={WORD_UNKNOWN: WORD_UNKNOWN_ID},
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


def char_emb_ids(word: str, embeddings_count, embedding_length=3):
    w = ' ' + word + ' '
    n = len(w)
    return [hash(w[i:i + n + 1]) % embeddings_count for i in range(n - embedding_length + 1)]


def lpad(arr, size, pad_value):
    """
    >>> lpad([], 2, -1)
    [-1, -1]
    >>> lpad([1, 2], 3, -1)
    [-1, 1, 2]
    >>> lpad([1, 2, 3], 2, -1)
    [2, 3]
    """
    assert size > 0
    given_arr = arr[-size:]
    to_add = size - len(given_arr)
    return [pad_value] * to_add + given_arr


def heads_to_childs(heads: Dict[int, int]) -> Dict[int, List[int]]:
    """
    >>> heads_to_childs({1:2, 2: 0, 3: 2}) == {2: [1, 3], 0: [2]}
    True
    """
    children = defaultdict(list)
    for c, h in heads.items():
        children[h].append(c)
    return children


def feat_children(target_node: int, heads: Dict[int, int], null_value=-1):
    """
    >>> feat_children(2, {1:2, 2: 0, 3: 2})
    [-1, 1, -1, 3]
    >>> feat_children(3, {1:2, 2: 0, 3: 2})
    [-1, -1, -1, -1]
    """
    children = heads_to_childs(heads)
    lc1 = sorted([c for c in children[target_node] if c < target_node])[-2:]
    rc1 = sorted([c for c in children[target_node] if c > target_node])[0:2]

    return lpad(lc1, 2, null_value) + lpad(rc1, 2, null_value)


# SyntaxState = namedtuple('SyntaxState', 'words buffer_index arcs arc_labels stack buffer')
SyntaxState = namedtuple('SyntaxState', 'words tags embeddings arcs stack buffer arc_labels')


class ArcStandardParser(nn.Module):
    def __init__(self, words_dict: Dict[str, int], tags_dict: Dict[str, int], dependency_relations_count):
        super().__init__()
        self.words_dict = words_dict
        self.tags_dict = tags_dict

        self.dependency_relations_count = dependency_relations_count
        self.words_count = len(self.words_dict)

        self.word_emb = nn.Embedding(self.words_count, WORD_DIM)
        self.word_emb.weight.data.normal_(0, np.sqrt(1/INPUT_DIM))

        self.tag_emb = nn.Embedding(TAG_EMB_COUNT, TAG_DIM)
        self.tag_emb.weight.data.normal_(0, np.sqrt(1/INPUT_DIM))

        self.char_emb = nn.EmbeddingBag(CHAR_EMB_COUNT, CHAR_DIM, mode='sum')
        self.char_emb.weight.data.normal_(0, np.sqrt(1/INPUT_DIM))

        self.input2hidden = nn.Linear(INPUT_DIM, HIDDEN_SIZE)
        self.hidden2output = nn.Linear(HIDDEN_SIZE, 1 + 2 * self.dependency_relations_count)

        self.set_device = None
        self.cpu()

    def cuda(self, device_id=None):
        super().cuda(device_id)
        self.set_device = lambda x: x.cuda(device_id)

    def cpu(self):
        super().cpu()
        self.set_device = lambda x: x.cpu()

    def _batch_char_emb(self, batch: List[List[str]]):
        word_map = [len(ws) for ws in batch]
        batch = list(chain(*[[char_emb_ids(w, CHAR_EMB_COUNT) for w in ws] for ws in batch]))
        offsets = [len(w) for w in batch]
        offsets.insert(0, 0)
        offsets.pop()
        embs = self.char_emb(self.set_device(Variable(torch.LongTensor(list(chain(*batch))))),
                             self.set_device(Variable(torch.cumsum(torch.LongTensor(offsets), 0))))
        return embs, np.cumsum([0] + word_map)

    def embed(self, sentences: List[List[str]], tags: List[List[str]]):
        words_batch = [self.words_dict.get(w, WORD_UNKNOWN_ID) for s in sentences for w in s]
        word_embs = self.word_emb(self.set_device(Variable(torch.LongTensor(words_batch))))

        char_batch = [char_emb_ids(w, CHAR_EMB_COUNT) for s in sentences for w in s]
        char_offsets = [len(w) for w in char_batch]
        char_offsets.insert(0, 0)
        char_offsets.pop()
        char_embs = self.char_emb(self.set_device(Variable(torch.LongTensor(list(chain(*char_batch))))),
                      self.set_device(Variable(torch.cumsum(torch.LongTensor(char_offsets), 0))))

        tags_batch = [self.tags_dict.get(t, WORD_UNKNOWN_ID) for ts in tags for t in ts]
        tag_embs = self.tag_emb(self.set_device(Variable(torch.LongTensor(tags_batch))))

        embs = torch.cat([word_embs, char_embs, tag_embs], 1)

        index = 0
        res = []
        for s in sentences:
            res.append(embs[index:index + len(s)])
            index += len(s)
        return res

    def create_initial_states(self, sentences: List[List[str]], tags: List[List[str]]):
        states = []
        sentences = [[WORD_ROOT] + sent + [WORD_EMPTY] * 3 for sent in sentences]
        tags = [[WORD_ROOT] + tag + [WORD_EMPTY] * 3 for tag in tags]
        embeddings = self.embed(sentences, tags)

        for ws, ts, embs in zip(sentences, tags, embeddings):
            word_empty_index = len(ws) - 1

            state = SyntaxState(ws, ts, embs, {}, [word_empty_index, word_empty_index, 0],
                                [b + 1 for b in range(len(ws) - 1)], {})
            states.append(state)
        return states

    def act(self, states, actions):
        new_states = []
        for s, a in zip(states, actions):
            action_id = 0 if a == 0 else (a - 1) // self.dependency_relations_count + 1
            dep_id = None if a == 0 else (a - 1) % self.dependency_relations_count
            if action_id == 0:  # shift
                if len(s.buffer) > 3:
                    # s.arcs,
                    # s.arc_labels,
                    ns = SyntaxState(s.words,
                                     s.tags,
                                     s.embeddings,
                                     s.arcs,
                                     s.stack + [s.buffer[0]],
                                     s.buffer[1:],
                                     s.arc_labels)
                else:
                    ns = s
            else:
                ok = True
                if action_id == 1:  # right-arc
                    if len(s.stack) > 3:
                        child = s.stack[-1]
                        head = s.stack[-2]
                    else:
                        ok = False

                elif action_id == 2:  # left-arc
                    if len(s.stack) > 4:
                        child = s.stack[-2]
                        head = s.stack[-1]
                    else:
                        ok = False

                else:
                    raise RuntimeError('Unknown action index')

                if ok:
                    new_arcs = dict(s.arcs)
                    new_arcs[child] = head

                    new_arc_labels = dict(s.arc_labels)
                    new_arc_labels[child] = dep_id

                    # new_arcs,
                    # new_arc_labels,
                    ns = SyntaxState(s.words,
                                     s.tags,
                                     s.embeddings,
                                     new_arcs,
                                     s.stack[:-2] + [head],
                                     s.buffer,
                                     new_arc_labels)
                else:
                    ns = s

            new_states.append(ns)
        return new_states

    def get_legal_actions(self, states: List[SyntaxState]):
        legal_actions = np.zeros([len(states), 1 + self.dependency_relations_count * 2]) + 1
        for i, s in enumerate(states):
            if len(s.buffer) > 3:
                legal_actions[i, 0] = 0
            if len(s.stack) <= 3:
                legal_actions[i, 1:self.dependency_relations_count + 1] = 0
            if len(s.stack) <= 4:
                legal_actions[i, self.dependency_relations_count + 1:] = 0
        return self.set_device(Variable(torch.FloatTensor(legal_actions)))

                # stack_len = self.set_device(torch.LongTensor([len(s.stack) for s in states]))
        # buff_len = self.set_device(torch.LongTensor([len(s.buffer) for s in states]))
        #
        # l0 = buff_len > 3
        # l2 = stack_len > 3
        # l1 = l0 * l2
        # return Variable(torch.stack([l0, l1, l2], dim=1).float())

    @staticmethod
    def terminated(states):
        return [len(s.buffer) == 3 and len(s.stack) == 2 for s in states]

    def forward(self, states: List[SyntaxState], training=False):
        # children_indexes = torch.LongTensor([feat_children(s.buffer_index, s.arcs, len(s.buffer) - 1) for s in states])

        stack_indexes = self.set_device(torch.LongTensor([s.stack[-3:] for s in states]))
        buffer_indexes = self.set_device(torch.LongTensor([s.buffer[:3] for s in states]))
        indexes = torch.cat([stack_indexes, buffer_indexes], dim=1)

        X = torch.stack([s.embeddings[indexes[i]].view(-1) for i, s in enumerate(states)])

        hid = F.dropout(F.relu(self.input2hidden(X)), p=0.5, training=training)
        out = self.hidden2output(hid)

        res = out
        return res, self.get_legal_actions(states)


def chain_head(head: int, child: int, heads: Dict[int, int]):
    """
    >>> chain_head(0, 2, {1: 2, 2: 3, 3: 0})
    True
    >>> chain_head(2, 0, {1: 2, 2: 3, 3: 0})
    False
    """
    curr_child = child
    while curr_child != -1:
        if curr_child == head:
            return True
        curr_child = heads.get(curr_child, -1)
    return False


def get_errors(stack: List[int], buffer: List[int], heads: Dict[int, int], punishment: int = PUNISH,
               shift_basic_error=SHIFT_BASIC_ERROR):
    """
    >>> get_errors([0], [1, 2, 3], {1: 2, 2: 3, 3: 0}, 10, 11)
    [0, 10, 10]
    >>> get_errors([0, 1], [2, 3], {1: 2, 2: 3, 3: 0}, 10, 11)
    [0, 1, 10]
    >>> get_errors([0, 1, 2], [3], {1: 2, 2: 3, 3: 0}, 10, 11)
    [11, 2, 0]
    >>> get_errors([0, 2], [3], {1: 2, 2: 3, 3: 0}, 10, 11)
    [0, 1, 10]
    >>> get_errors([0, 2, 3], [], {1: 2, 2: 3, 3: 0}, 10, 11)
    [10, 2, 0]
    >>> get_errors([0, 3], [], {1: 2, 2: 3, 3: 0}, 10, 11)
    [10, 0, 10]
    >>> get_errors([0, 1, 2], [3], {1: 2, 2: 0, 3: 2}, 10, 11)
    [0, 3, 0]
    """
    if len(stack) < 2:
        return [0, punishment, punishment]
    rword = stack[-1]
    lword = stack[-2]

    r_err = len([w for w in chain(stack, buffer) if heads.get(w, -1) == rword])
    if heads[rword] != lword and heads[rword] in chain(stack, buffer):
        r_err += 1

    if len(stack) < 3:
        l_err = punishment
    else:
        l_err = len([w for w in chain(stack, buffer) if heads.get(w, -1) == lword])
        if heads[lword] != rword and heads[lword] in chain(stack, buffer):
            l_err += 1

    if not buffer:
        s_err = punishment
    else:
        if chain_head(rword, buffer[0], heads):
            s_err = 0
        elif heads[rword] == buffer[0] and rword not in [heads.get(w, -1) for w in chain(stack, buffer)]:
            s_err = 0
        elif heads.get(lword, -1) == rword:
            s_err = shift_basic_error
        elif heads.get(rword, -1) == lword and rword not in [heads.get(w, -1) for w in chain(stack, buffer)]:
            s_err = shift_basic_error
        elif heads[buffer[0]] in stack:
            s_err = shift_basic_error
        else:
            s_err = 0

    return [s_err, r_err, l_err]


def get_labels_errors(stack: List[int], errors: List[int], labels: Dict[int, int], labels_count: int):
    r_errors = [errors[1] + (0.5 if labels.get(stack[-1], -1) != l else 0) for l in range(labels_count)]
    l_errors = [errors[2] + (0.5 if labels.get(stack[-2], -1) != l else 0) for l in range(labels_count)]

    return [errors[0]] + r_errors + l_errors


##################################   RUN TEST  ####################################

import doctest

doctest.testmod()

# print('EARLY EXIT', file=sys.stderr)
# sys.exit().get(stack[-2], -1)

##################################   TRAINING   ###################################

train, test, dictionary, tags_dictionary, deprel_dictionary = cached('downloads/ud_arcstandard.pickle',
                                                                     lambda: prepare_data('ud', 'english'))

parser = ArcStandardParser(dictionary, tags_dictionary, len(deprel_dictionary))

device_id = int(os.getenv('PYTORCH_GPU_ID', -1))
if device_id >= 0:
    parser.cuda(device_id)

print('Using cuda device "{}"'.format(device_id))

optimizer = Adam(parser.parameters(), LR, betas=(0.9, 0.9), weight_decay=L2_DECAY)

criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()

seen_samples = 0
losses = []

times = []
for batch in batch_generator(train, BATCH_SIZE):
    start = time()
    seen_samples += len(batch)

    batch = list(deepcopy(batch))

    loss = parser.set_device(Variable(torch.zeros(1)))

    ids, sents, tag_ids, tags, heads, deprels = zip(*[list(zip(*sent)) for sent in batch])
    heads = [{i + 1: h for i, h in enumerate(head)} for head in heads]
    deprels = [{i + 1: r for i, r in enumerate(rel)} for rel in deprels]
    sents = [list(ws) for ws in sents]
    ids = [list(ws) for ws in ids]
    tags = [list(ws) for ws in tags]
    tag_ids = [list(ws) for ws in tag_ids]

    states = parser.create_initial_states(sents, tags)

    correct_actions = 0
    total_actions = 0

    correct_heads = 0
    correct_rels = 0
    total_heads = 0

    example = []
    while states:
        decisions, legal_actions = parser.forward(states, training=True)
        # decisions /= decisions + 1

        errors = [get_labels_errors(s.stack,
                                    get_errors(s.stack[2:], s.buffer[:-3], h),
                                    labels=r,
                                    labels_count=parser.dependency_relations_count
                                    ) for s, h, r in
                  zip(states, heads, deprels)]
        # if [e for e in errors if min(e)]:
        #     raise RuntimeError('Bugs!!!')
        # ys = [e.index(min(e)) for e in errors]
        errors = torch.FloatTensor(errors)
        # for y, e in zip(ys, errors):
        #     assert e[y] == 0
        rights = Variable(parser.set_device((errors - errors.min(1, keepdim=True)[0] == 0).float()))


        local_loss = criterion(decisions, rights)

        loss += local_loss * len(states)
        total_actions += len(states)

        _, argmax = ((decisions - decisions.min(1, keepdim=True)[0] + 1) * legal_actions).max(1)

        # correct_actions += (argmax == parser.set_device(Variable(torch.LongTensor(ys)))).long().sum().data[0]

        # example.append((states[0].words[states[0].index], ys[0], argmax.data[0]))

        # states = parser.act(states, ys)
        states = parser.act(states, argmax.data.tolist())

        terminated = ArcStandardParser.terminated(states)

        for i in reversed(range(len(states))):
            if terminated[i]:
                total_heads += len(heads[i])
                for w in range(len(heads[i])):
                    if states[i].arcs[w + 1] == heads[i][w + 1]:
                        correct_heads += 1
                        if states[i].arc_labels[w + 1] == deprels[i][w + 1]:
                            correct_rels += 1
                states.pop(i)
                # batch_ga.pop(i)
                heads.pop(i)
                deprels.pop(i)

                # optimizer.zero_grad()
                # local_loss.backward(retain_graph=bool(states))
                # optimizer.step()

    assert not states

    loss = loss / total_actions
    optimizer.zero_grad()
    loss.backward()
    losses.append(loss.data[0])

    optimizer.step()

    times.append(time() - start)
    # print(example)
    print('{}'.format(seen_samples).ljust(8),
          # '{:.1f}%'.format(correct_actions / total_actions * 100),
          '{:.1f}%'.format(correct_heads / total_heads * 100),
          '{:.1f}%'.format(correct_rels / total_heads * 100),
          np.mean(losses[-10:]),
          '{:.3f}s'.format(sum(times) / len(times)),
          sep='\t')

    if (seen_samples // BATCH_SIZE) % (TEST_EVERY_SAMPLES // BATCH_SIZE) == 0:
        test_started = time()

        batch = list(deepcopy(test))

        ids, sents, tag_ids, tags, heads, deprels = zip(*[list(zip(*sent)) for sent in batch])
        heads = list(heads)
        deprels = list(deprels)
        sents = [list(ws) for ws in sents]
        ids = [list(ws) for ws in ids]
        tag_ids = [list(ws) for ws in tag_ids]

        states = parser.create_initial_states(sents, ids, tag_ids)

        correct_actions = 0
        total_actions = 0

        correct_heads = 0
        correct_rels = 0
        total_heads = 0

        example = []
        while states:
            decisions, legal_actions = parser.forward(states)
            decisions -= decisions.min(1, keepdim=True)[0] - 1
            decisions = decisions * legal_actions

            total_actions += len(states)

            _, argmax = decisions.max(1)
            states = parser.act(states, argmax.data.cpu().tolist())

            terminated = ArcStandardParser.terminated(states)

            for i in reversed(range(len(terminated))):
                if terminated[i]:
                    total_heads += len(heads[i])
                    for w in range(len(heads[i])):
                        if states[i].arcs[w + 1] == heads[i][w]:
                            correct_heads += 1
                            if states[i].arc_labels[w + 1] == deprels[i][w]:
                                correct_rels += 1
                    states.pop(i)
                    heads.pop(i)
                    deprels.pop(i)

        assert not states

        test_duration = (time() - test_started)
        test_total_tokens_count = sum([len(s) for s in test])
        token_per_sec = test_total_tokens_count / test_duration
        print('TEST', '{}'.format(len(test)).ljust(8),
              '{:.1f}%'.format(correct_heads / total_heads * 100),
              '{:.1f}%'.format(correct_rels / total_heads * 100),
              '{:.1f} w/s'.format(token_per_sec),
              sep='\t')

    if seen_samples > STOP_AFTER_SAMPLES:
        break
