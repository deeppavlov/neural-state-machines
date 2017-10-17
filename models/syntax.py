from collections import namedtuple
from itertools import chain
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np

from data_providers.ud_pos import pos as ud

conllu = ud.DataProvider(lang='english')
# conllu = ud.DataProvider(lang='russian')
train = []
for s in conllu.train:
    try:
        sent = []
        for w in s:
            sent.append((int(w.id), w.form, int(w.head)))
        train.append(sent)
    except ValueError:
        pass

test = []
for s in conllu.dev:
    try:
        sent = []
        for w in s:
            sent.append((int(w.id), w.form, int(w.head)))
        test.append(sent)
    except ValueError:
        pass

SyntaxState = namedtuple('SyntaxState', 'words buffer_index arcs stack buffer')

# POSState = namedtuple('POSState', 'chars words index outputs')
#
# STOP_AFTER_SAMPLES = 10 * 1000
# TEST_EVERY_SAMPLES = 6000
CHAR_EMB_COUNT = 500
WORD_EMB_COUNT = 30 * 1000
HIDDEN_SIZE = 50
LR = 0.01
BATCH_SIZE = 29

WORD_ROOT = ' '
WORD_EMPTY = '\t'


def char_emb_ids(word: str, embeddings_count, embedding_length=3):
    w = ' ' + word + ' '
    n = len(w)
    return [hash(w[i:i + n + 1]) % embeddings_count for i in range(n - embedding_length + 1)]


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
                ns = SyntaxState(s.words,
                                 s.buffer_index + 1,
                                 s.arcs,
                                 s.stack + [(s.buffer_index + 1, s.buffer[s.buffer_index])],
                                 s.buffer)
            else:
                if a == 1:  # right-arc
                    child = s.stack[-2]
                    head = s.stack[-1]

                elif a == 2:  # left-arc
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

    def forward(self, states: List[SyntaxState]):
        buffers = []
        stacks = []
        for s in states:
            buffers.append(s.buffer[s.buffer_index: s.buffer_index + 3].view(-1))
            stacks.append(torch.cat([st[1] for st in s.stack[-3:]]))

        buffers = torch.stack(buffers)
        stacks = torch.stack(stacks)
        X = torch.cat([buffers, stacks], dim=1)
        res = self.linear(X)
        res -= res.max(1, keepdim=True)[0]
        return res.exp()


tbsp = TBSyntaxParser()

# data = [[w[1] for w in s] for s in train]
#
# batch = data[:3]
#
# states = tbsp.create_initial_states(batch)
#
# res = tbsp.forward(states)
# res = res / res.sum(1, keepdim=True)
# pass


def batch_generator(seq: List, batch_size):
    # while True:
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


optimizer = Adam(tbsp.parameters(), LR)
criterion = nn.CrossEntropyLoss()

seen_samples = 0
losses = []

sents_count = len(train)
errors_count = 0
for batch in batch_generator(train, BATCH_SIZE):
    seen_samples += len(batch)

    loss = Variable(torch.zeros(1))

    # ids, sents, heads = zip(*[list(zip(*sent)) for sent in batch])
    # sents = [list(ws) for ws in sents]

    ids = []
    sents = []
    heads = []
    gold = []
    for sent in batch:
        i, ws, h = zip(*sent)
        try:
            gold.append(gold_actions(h))
            ids.append(i)
            sents.append(list(ws))
            heads.append(h)
        except RuntimeError as e:
            errors_count += 1
            # print()
            # print(e)
            # for i, h, ws in zip(i, h, ws):
            #     print(i, h, ws)

    states = tbsp.create_initial_states(sents)

    state = states[0]
    g = gold[0]
    for g in g:
        state = tbsp.act([state], [g])[0]

    assert list(state.arcs.values()) == list(heads[0])

    # 0/0

print('{:.2f}% errors ({} out of {})'.format(100 * errors_count/sents_count, errors_count, sents_count))

    # batch_tags = [list(tags) for tags in batch_tags]
    # states = pos_model.create_initial_states(inputs)
    #
    # state_max_len = max([len(s.chars)-2 for s in states])
    # words_seen = 0
    # correct_words = 0
    # example = []
    # for i in range(state_max_len):
    #     decisions = pos_model.forward(states)
    #     decisions /= decisions.sum(1, keepdim=True)
    #     ys = [tagmap[tag.pop(0)] for tag in batch_tags]
    #     loss += criterion(decisions, Variable(torch.LongTensor(ys))) * len(states)
    #     words_seen += len(states)
    #
    #     _, argmax = decisions.max(1)
    #     correct_words += (argmax == Variable(torch.LongTensor(ys))).long().sum().data[0]
    #
    #     example.append((states[0].words[states[0].index], id2tag[ys[0]], id2tag[argmax.data[0]]))
    #
    #     new_states = pos_model.act(states, ys)
    #
    #     states = [s for s in new_states if s.index < len(s.chars) - 1]
    #     batch_tags = [tag for tag in batch_tags if tag]
    #
    # assert not states
    #
    # loss = loss / words_seen
    # optimizer.zero_grad()
    # loss.backward()
    # losses.append(loss.data[0])
    #
    # optimizer.step()
    #
    # # print(example)
    # print('{:.2f}'.format(seen_samples).ljust(8), '{:.1f}%'.format(correct_words/words_seen*100), np.mean(losses[-10:]), sep='\t')
    #
    # if (seen_samples // BATCH_SIZE) % (TEST_EVERY_SAMPLES // BATCH_SIZE) == 0:
    #     inputs, test_tags = zip(*test_sents)
    #     test_tags = [list(tags) for tags in test_tags]
    #     states = pos_model.create_initial_states(inputs)
    #
    #     state_max_len = max([len(s.chars) - 2 for s in states])
    #     words_seen = 0
    #     correct_words = 0
    #     loss = 0
    #     example = []
    #     for i in range(state_max_len):
    #         decisions = pos_model.forward(states)
    #         decisions /= decisions.sum(1, keepdim=True)
    #         ys = [tagmap[tag.pop(0)] for tag in test_tags]
    #         loss += criterion(decisions, Variable(torch.LongTensor(ys))).data[0] * len(states)
    #         words_seen += len(states)
    #
    #         _, argmax = decisions.max(1)
    #         correct_words += (argmax == Variable(torch.LongTensor(ys))).long().sum().data[0]
    #
    #         example.append((states[0].words[states[0].index], id2tag[ys[0]], id2tag[argmax.data[0]]))
    #
    #         new_states = pos_model.act(states, argmax.data.tolist())
    #
    #         states = [s for s in new_states if s.index < len(s.chars) - 1]
    #         test_tags = [tag for tag in test_tags if tag]
    #     assert not states
    #     loss /= words_seen
    #     print('Test'.ljust(8), '{:.1f}%'.format(correct_words/words_seen*100), '{:.2f}'.format(loss).ljust(8), sep='\t')
    #     print('Test'.ljust(8), example)













