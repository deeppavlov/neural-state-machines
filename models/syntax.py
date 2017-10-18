from collections import namedtuple
from itertools import chain
from typing import List, Iterator
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np

from copy import deepcopy

from data_providers.ud_pos import pos as ud


STOP_AFTER_SAMPLES = 2 * 1000
TEST_EVERY_SAMPLES = 200
CHAR_EMB_COUNT = 500
WORD_EMB_COUNT = 30 * 1000
HIDDEN_SIZE = 50
LR = 0.01
BATCH_SIZE = 29

WORD_ROOT = ' '
WORD_EMPTY = '\t'


def batch_generator(seq: Iterator, batch_size):
    seq = list(seq)
    while True:
        indexi = torch.randperm(len(seq))
        for i in range(0, len(indexi), batch_size):
            yield [seq[k] for k in indexi[i:i+batch_size]]


def gold_actions(heads: List[int]):
    """
    >>> gold_actions([3, 3, 0])
    [0, 0, 0, 1, 1, 2]
    >>> gold_actions([2, 0, 6, 6, 6, 2, 2])
    [0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 2, 0, 2, 2]
    """
    w2h = {i: h for i, h in zip(range(1, len(heads)+1), heads)}
    stack = [0, 1]
    buffer = list(range(2, len(heads)+1))
    actions = [0]

    while buffer or len(stack) > 1:
        if len(stack) < 2:
            # shift
            actions.append(0)
            stack.append(buffer.pop(0))
        elif (stack[-1] not in w2h.values()) and w2h[stack[-1]] == stack[-2]:
            # left-arc
            actions.append(2)
            del w2h[stack[-1]]
            stack.pop(-1)
        elif (stack[-2] not in w2h.values()) and w2h[stack[-2]] == stack[-1]:
            # right-arc
            actions.append(1)
            del w2h[stack[-2]]
            stack.pop(-2)
        else:
            if not buffer:
                raise RuntimeError('Wrong sentence markup')
            # shift
            actions.append(0)
            stack.append(buffer.pop(0))

    return actions


conllu = ud.DataProvider(lang='english')
# conllu = ud.DataProvider(lang='russian')
train = []
train_ga = []
train_gold_errors = 0
for s in conllu.train:
    try:
        sent = []
        for w in s:
            sent.append((int(w.id), w.form, int(w.head)))
        ga = gold_actions([e[2] for e in sent])
        train.append(sent)
        train_ga.append(ga)
    except ValueError:
        pass
    except RuntimeError:
        train_gold_errors += 1

test = []
test_ga = []
test_gold_errors = 0
for s in conllu.dev:
    try:
        sent = []
        for w in s:
            sent.append((int(w.id), w.form, int(w.head)))
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
        self.linear = nn.Linear(HIDDEN_SIZE*6, 3)

    def _batch_char_emb(self, batch: List[List[str]]):
        word_map = [len(ws) for ws in batch]
        batch = list(chain(*[[char_emb_ids(w, CHAR_EMB_COUNT) for w in ws] for ws in batch]))
        offsets = [len(w) for w in batch]
        offsets.insert(0, 0)
        offsets.pop()
        embs = self.char_emb(Variable(torch.LongTensor(list(chain(*batch)))),
                             Variable(torch.cumsum(torch.LongTensor(offsets), 0)))
        return embs, np.cumsum([0] + word_map)

    def create_initial_states(self, sentences: List[List[str]]):
        states = []
        sentences = [[WORD_ROOT] + sent + [WORD_EMPTY, WORD_EMPTY, WORD_EMPTY] for sent in sentences]

        embs, word_indexes = self._batch_char_emb(sentences)
        for i, ws in enumerate(sentences):
            buffer = embs[word_indexes[i]:word_indexes[i + 1]]

            state = SyntaxState(ws[1:-3], 0, {}, [(-1, buffer[-1]), (-1, buffer[-1]), (0, buffer[0])], buffer[1:])
            states.append(state)
        return states

    def act(self, states, actions):
        new_states = []
        for s, a in zip(states, actions):
            if a == 0:  # shift
                assert s.buffer_index + 3 < len(s.buffer)
                ns = SyntaxState(s.words,
                                 s.buffer_index + 1,
                                 s.arcs,
                                 s.stack + [(s.buffer_index + 1, s.buffer[s.buffer_index])],
                                 s.buffer)
            else:
                if a == 1:  # right-arc
                    assert len(s.stack) > 4
                    child = s.stack[-2]
                    head = s.stack[-1]

                elif a == 2:  # left-arc
                    assert len(s.stack) > 3
                    child = s.stack[-1]
                    head = s.stack[-2]

                else:
                    raise RuntimeError('Unknown action index')

                new_arcs = dict(s.arcs)
                new_arcs[child[0]] = head[0]

                ns = SyntaxState(s.words,
                                 s.buffer_index,
                                 new_arcs,
                                 s.stack[:-2] + [head],
                                 s.buffer)

            new_states.append(ns)
        return new_states

    @staticmethod
    def terminated(states):
        return [s.buffer_index + 3 == len(s.buffer) and len(s.stack) == 3 for s in states]

    def forward(self, states: List[SyntaxState]):
        buffers = []
        stacks = []
        legal_actions = Variable(torch.zeros(len(states), 3) + 1)
        for i, s in enumerate(states):
            buffers.append(s.buffer[s.buffer_index: s.buffer_index + 3].view(-1))
            stacks.append(torch.cat([st[1] for st in s.stack[-3:]]))
            if s.buffer_index + 3 >= len(s.buffer):
                legal_actions[i, 0] = 0
            if len(s.stack) <= 4:
                legal_actions[i, 1] = 0
            if len(s.stack) <= 3:
                legal_actions[i, 2] = 0

        buffers = torch.stack(buffers)
        stacks = torch.stack(stacks)
        X = torch.cat([buffers, stacks], dim=1)
        res = self.linear(X)
        res -= res.max(1, keepdim=True)[0]
        return res.exp() * legal_actions


parser = TBSyntaxParser()

# data = [[w[1] for w in s] for s in train]
#
# batch = data[:3]
#
# states = tbsp.create_initial_states(batch)
#
# res = tbsp.forward(states)
# res = res / res.sum(1, keepdim=True)
# pass


optimizer = Adam(parser.parameters(), LR)
criterion = nn.CrossEntropyLoss()

seen_samples = 0
losses = []

for batch in batch_generator(zip(train, train_ga), BATCH_SIZE):
    batch, batch_ga = zip(*batch)
    seen_samples += len(batch)

    batch = list(deepcopy(batch))
    batch_ga = list(deepcopy(batch_ga))

    loss = Variable(torch.zeros(1))

    ids, sents, heads = zip(*[list(zip(*sent)) for sent in batch])
    heads = list(heads)
    sents = [list(ws) for ws in sents]

    states = parser.create_initial_states(sents)

    correct_actions = 0
    total_actions = 0

    correct_heads = 0
    total_heads = 0

    example = []
    while states:
        decisions = parser.forward(states)
        decisions /= decisions.sum(1, keepdim=True)
        ys = [s.pop(0) for s in batch_ga]
        loss += criterion(decisions, Variable(torch.LongTensor(ys))) * len(states)
        total_actions += len(states)

        _, argmax = decisions.max(1)
        correct_actions += (argmax == Variable(torch.LongTensor(ys))).long().sum().data[0]

        # example.append((states[0].words[states[0].index], ys[0], argmax.data[0]))

        states = parser.act(states, ys)

        terminated = TBSyntaxParser.terminated(states)

        for i in reversed(range(len(batch_ga))):
            if not batch_ga[i]:
                assert terminated[i]
                total_heads += len(heads[i])
                for w in range(len(heads[i])):
                    if states[i].arcs[w+1] == heads[i][w]:
                        correct_heads += 1
                states.pop(i)
                batch_ga.pop(i)
                heads.pop(i)

    assert not states

    loss = loss / total_actions
    optimizer.zero_grad()
    loss.backward()
    losses.append(loss.data[0])

    optimizer.step()

    # print(example)
    print('{:.2f}'.format(seen_samples).ljust(8), '{:.1f}%'.format(correct_actions / total_actions * 100), np.mean(losses[-10:]), sep='\t')

    if (seen_samples // BATCH_SIZE) % (TEST_EVERY_SAMPLES // BATCH_SIZE) == 0:

        batch = list(deepcopy(test))
        batch_ga = list(deepcopy(test_ga))

        loss = Variable(torch.zeros(1))

        ids, sents, heads = zip(*[list(zip(*sent)) for sent in batch])
        heads = list(heads)
        sents = [list(ws) for ws in sents]

        states = parser.create_initial_states(sents)

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

            # example.append((states[0].words[states[0].index], ys[0], argmax.data[0]))

            states = parser.act(states, argmax.data.tolist())

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

        print('TEST', '{}'.format(len(test)).ljust(8),
              '{:.1f}%'.format(correct_heads / total_heads * 100),
              sep='\t')

    if seen_samples > STOP_AFTER_SAMPLES:
        break
