import random
from collections import namedtuple, defaultdict
from itertools import chain
from time import time
from typing import List, Iterator, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import os

from copy import deepcopy

from data_providers.ud_pos import pos as ud

STOP_AFTER_SAMPLES = 200 * 1000
TEST_EVERY_SAMPLES = 2000
CHAR_EMB_COUNT = 5000
WORD_EMB_COUNT = 100 * 1000
HIDDEN_SIZE = 50
LR = 0.001
BATCH_SIZE = 256

WORD_ROOT = ' '
WORD_EMPTY = '\t'
WORD_EMPTY_ID = 2
WORD_ROOT_ID = 1
WORD_UNKNOWN_ID = 0

BUCKETS_COUNT = 8

PUNISH = 10


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


def gold_actions(heads: List[int]):
    """
    >>> gold_actions([3, 3, 0])
    [0, 0, 0, 1, 1, 2]
    >>> gold_actions([2, 0, 6, 6, 6, 2, 2])
    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 2, 0, 2, 2]
    """
    w2h = {i: h for i, h in zip(range(1, len(heads) + 1), heads)}
    stack = [0, 1]
    buffer = list(range(2, len(heads) + 1))
    actions = [0]

    while buffer or len(stack) > 1:
        if len(stack) < 2:
            # shift
            actions.append(0)
            stack.append(buffer.pop(0))
        elif (stack[-1] not in w2h.values()) and w2h[stack[-1]] == stack[-2]:
            # right-arc
            actions.append(1)
            del w2h[stack[-1]]
            stack.pop(-1)
        elif (stack[-2] not in w2h.values()) and w2h[stack[-2]] == stack[-1]:
            # left-arc
            actions.append(2)
            del w2h[stack[-2]]
            stack.pop(-2)
        else:
            if not buffer:
                raise RuntimeError('Wrong sentence markup')
            # shift
            actions.append(0)
            stack.append(buffer.pop(0))

    return actions


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

conllu = ud.DataProvider(lang='english')
# conllu = ud.DataProvider(lang='russian')

dictionary = create_dictionary(chain(*([w.form for w in s] for s in conllu.train)),
                               reserved_ids={'_UKNOWN_': WORD_UNKNOWN_ID, WORD_ROOT: WORD_ROOT_ID,
                                             WORD_EMPTY: WORD_EMPTY_ID})
print('Dictionary has {} elements'.format(len(dictionary)))

train = []
train_ga = []
train_gold_errors = 0
for s in conllu.train:
    try:
        sent = []
        for w in s:
            int(w.id)  # crash if not integer
            sent.append((dictionary.get(w.form, WORD_UNKNOWN_ID), w.form, int(w.head)))
        ga = gold_actions([e[2] for e in sent])
        train.append(sent)
        train_ga.append(ga)
    except ValueError:
        pass
    except RuntimeError:
        train_gold_errors += 1
print('Train has {} examples'.format(len(train)))

test = []
test_ga = []
test_gold_errors = 0
for s in conllu.dev:
    try:
        sent = []
        for w in s:
            int(w.id)
            sent.append((dictionary.get(w.form, WORD_UNKNOWN_ID), w.form, int(w.head)))
        ga = gold_actions([e[2] for e in sent])
        test.append(sent)
        train_ga.append(ga)
    except ValueError:
        pass
    except RuntimeError:
        test_gold_errors += 1


def char_emb_ids(word: str, embeddings_count, embedding_length=3):
    w = ' ' + word + ' '
    n = len(w)
    return [hash(w[i:i + n + 1]) % embeddings_count for i in range(n - embedding_length + 1)]


SyntaxState = namedtuple('SyntaxState', 'words buffer_index arcs stack buffer')


class TBSyntaxParser(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_emb = nn.Embedding(WORD_EMB_COUNT, HIDDEN_SIZE)
        self.char_emb = nn.EmbeddingBag(CHAR_EMB_COUNT, HIDDEN_SIZE, mode='sum')
        self.linear = nn.Linear(HIDDEN_SIZE * 6, 3)

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

    def create_initial_states(self, sentences: List[List[str]], ids: List[List[int]]):
        states = []
        sentences = [[WORD_ROOT] + sent + [WORD_EMPTY] * 3 for sent in sentences]
        id_sents = [[WORD_ROOT_ID] + sent + [WORD_EMPTY_ID] * 3 for sent in ids]

        char_embs, word_indexes = self._batch_char_emb(sentences)

        for i, (ws, ids) in enumerate(zip(sentences, id_sents)):
            word_empty_index = len(ws) - 1

            words_embeddings = self.word_emb(self.set_device(Variable(torch.LongTensor(ids))))
            chars_embeddings = char_embs[word_indexes[i]:word_indexes[i + 1]]
            # buffer = torch.cat((words_embeddings, chars_embeddings), dim=1)
            buffer = words_embeddings + chars_embeddings

            state = SyntaxState(ws[1:-3], 1, {}, [word_empty_index, word_empty_index, 0], buffer)
            states.append(state)
        return states

    def act(self, states, actions):
        new_states = []
        for s, a in zip(states, actions):
            if a == 0:  # shift
                if s.buffer_index + 3 < len(s.buffer):
                    ns = SyntaxState(s.words,
                                     s.buffer_index + 1,
                                     s.arcs,
                                     s.stack + [s.buffer_index],
                                     s.buffer)
                else:
                    ns = s
            else:
                ok = True
                if a == 1:  # right-arc
                    if len(s.stack) > 3:
                        child = s.stack[-1]
                        head = s.stack[-2]
                    else:
                        ok = False

                elif a == 2:  # left-arc
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

                    ns = SyntaxState(s.words,
                                     s.buffer_index,
                                     new_arcs,
                                     s.stack[:-2] + [head],
                                     s.buffer)
                else:
                    ns = s

            new_states.append(ns)
        return new_states

    @staticmethod
    def terminated(states):
        return [s.buffer_index + 3 == len(s.buffer) and len(s.stack) == 3 for s in states]

    def forward(self, states: List[SyntaxState], test_mode=True):
        buffers = []
        stacks = []
        legal_actions = np.zeros([len(states), 3]) + 1
        stack_indexes = self.set_device(torch.LongTensor([s.stack[-3:] for s in states]))
        for i, s in enumerate(states):
            buffers.append(s.buffer[s.buffer_index: s.buffer_index + 3].view(-1))
            stacks.append(s.buffer[stack_indexes[i]].view(-1))
            if s.buffer_index + 3 >= len(s.buffer):
                legal_actions[i, 0] = 0
            if len(s.stack) <= 4:
                legal_actions[i, 2] = 0
            if len(s.stack) <= 3:
                legal_actions[i, 1] = 0

        buffers = torch.stack(buffers)
        stacks = torch.stack(stacks)
        X = torch.cat([buffers, stacks], dim=1)
        res = self.linear(X)
        res = torch.clamp(res, -10e5, 10)
        # res -= res.max(1, keepdim=True)[0]
        res = res.exp()
        if test_mode:
            res = res * self.set_device(Variable(torch.FloatTensor(legal_actions)))
        return res


def get_errors(stack: List[int], buffer: List[int], heads: Dict[int, int]):
    if len(stack) < 2:
        return [0, PUNISH, PUNISH]
    rword = stack[-1]
    lword = stack[-2]

    r_err = len([w for w in chain(stack, buffer) if heads.get(w, -1) == rword])
    if heads[rword] != lword and heads[rword] in chain(stack, buffer):
        r_err += 1

    if len(stack) < 3:
        l_err = PUNISH
    else:
        l_err = len([w for w in chain(stack, buffer) if heads.get(w, -1) == lword])
        if heads[lword] != rword and heads[lword] in chain(stack, buffer):
            l_err += 1

    if not buffer:
        s_err = PUNISH
    elif heads[buffer[0]] == rword or not [w for w in chain(stack, buffer) if heads.get(w, -1) == buffer[0]]:
        s_err = 0
    else:
        s_err = min(get_errors(stack + [buffer[0]], buffer[1:], heads))

    return [s_err, r_err, l_err]


assert get_errors([0], [1, 2, 3], {1: 2, 2: 3, 3: 0}) == [0, PUNISH, PUNISH]
assert get_errors([0, 1], [2, 3], {1: 2, 2: 3, 3: 0}) == [0, 1, PUNISH]
assert get_errors([0, 1, 2], [3], {1: 2, 2: 3, 3: 0}) == [1, 2, 0]
assert get_errors([0, 2], [3], {1: 2, 2: 3, 3: 0}) == [0, 1, PUNISH]
assert get_errors([0, 2, 3], [], {1: 2, 2: 3, 3: 0}) == [PUNISH, 2, 0]
assert get_errors([0, 3], [], {1: 2, 2: 3, 3: 0}) == [PUNISH, 0, PUNISH]
assert get_errors([0, 1, 2], [3], {1: 2, 2: 0, 3: 2}) == [0, 3, 0]

parser = TBSyntaxParser()

device_id = int(os.getenv('PYTORCH_GPU_ID', -1))
if device_id >= 0:
    parser.cuda(device_id)

optimizer = Adam(parser.parameters(), LR, betas=(0.9, 0.9))

criterion = nn.BCELoss()
# criterion = nn.CrossEntropyLoss()

seen_samples = 0
losses = []

times = []
for batch in batch_generator(zip(train, train_ga), BATCH_SIZE):
    start = time()
    batch, batch_ga = zip(*batch)
    seen_samples += len(batch)

    batch = list(deepcopy(batch))
    # batch_ga = list(deepcopy(batch_ga))

    loss = parser.set_device(Variable(torch.zeros(1)))

    ids, sents, heads = zip(*[list(zip(*sent)) for sent in batch])
    heads = [{i+1: h for i, h in enumerate(head)} for head in heads]
    sents = [list(ws) for ws in sents]
    ids = [list(ws) for ws in ids]

    states = parser.create_initial_states(sents, ids)

    correct_actions = 0
    total_actions = 0

    correct_heads = 0
    total_heads = 0

    example = []
    while states:
        decisions = parser.forward(states, test_mode=False)
        decisions /= decisions.sum(1, keepdim=True)

        # ys = [s.pop(0) for s in batch_ga]

        errors = [get_errors(s.stack[2:], list(range(s.buffer_index, len(s.buffer) - 3)), h) for s, h in zip(states, heads)]
        errors = torch.LongTensor(errors)
        # for y, e in zip(ys, errors):
        #     assert e[y] == 0
        rights = Variable(parser.set_device((errors - errors.min() == 0).float()))

        local_loss = criterion(decisions, rights)

        optimizer.zero_grad()
        local_loss.backward(retain_graph=bool(states))
        optimizer.step()

        loss += local_loss * len(states)
        total_actions += len(states)

        _, argmax = decisions.max(1)
        # correct_actions += (argmax == parser.set_device(Variable(torch.LongTensor(ys)))).long().sum().data[0]

        # example.append((states[0].words[states[0].index], ys[0], argmax.data[0]))

        # states = parser.act(states, ys)
        states = parser.act(states, argmax.data.tolist())

        terminated = TBSyntaxParser.terminated(states)

        for i in reversed(range(len(states))):
            if terminated[i]:
                total_heads += len(heads[i])
                for w in range(len(heads[i])):
                    if states[i].arcs[w + 1] == heads[i][w+1]:
                        correct_heads += 1
                states.pop(i)
                # batch_ga.pop(i)
                heads.pop(i)

    assert not states

    loss = loss / total_actions
    # optimizer.zero_grad()
    # loss.backward()
    losses.append(loss.data[0])

    # optimizer.step()

    times.append(time() - start)
    # print(example)
    print('{}'.format(seen_samples).ljust(8),
          # '{:.1f}%'.format(correct_actions / total_actions * 100),
          '{:.1f}%'.format(correct_heads / total_heads * 100),
          np.mean(losses[-10:]),
          '{:.3f}s'.format(sum(times) / len(times)),
          sep='\t')

    if (seen_samples // BATCH_SIZE) % (TEST_EVERY_SAMPLES // BATCH_SIZE) == 0:
        test_started = time()

        batch = list(deepcopy(test))
        batch_ga = list(deepcopy(test_ga))

        ids, sents, heads = zip(*[list(zip(*sent)) for sent in batch])
        heads = list(heads)
        sents = [list(ws) for ws in sents]
        ids = [list(ws) for ws in ids]

        states = parser.create_initial_states(sents, ids)

        correct_actions = 0
        total_actions = 0

        correct_heads = 0
        total_heads = 0

        example = []
        while states:
            decisions = parser.forward(states)
            decisions /= decisions.sum(1, keepdim=True)
            total_actions += len(states)

            _, argmax = decisions.max(1)
            states = parser.act(states, argmax.data.cpu().tolist())

            terminated = TBSyntaxParser.terminated(states)

            for i in reversed(range(len(terminated))):
                if terminated[i]:
                    total_heads += len(heads[i])
                    for w in range(len(heads[i])):
                        if states[i].arcs[w + 1] == heads[i][w]:
                            correct_heads += 1
                    states.pop(i)
                    heads.pop(i)

        assert not states

        test_duration = (time() - test_started)
        test_total_tokens_count = sum([len(s) for s in test])
        token_per_sec = test_total_tokens_count / test_duration
        print('TEST', '{}'.format(len(test)).ljust(8),
              '{:.1f}%'.format(correct_heads / total_heads * 100),
              '{:.1f} w/s'.format(token_per_sec),
              sep='\t')

    if seen_samples > STOP_AFTER_SAMPLES:
        break
