from collections import namedtuple
from itertools import chain
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np

import data_providers.ud_pos.pos as ud
import data_providers.gikrya.gikrya as gikrya

TAG_START = 'START'
TAG_END = 'END'


class NeuSM:
    def act(self, states, actions: List[int]) -> 'state':
        pass

    def forward(self, states) -> torch.FloatTensor:
        pass

    def create_initial_states(self, inputs):
        pass


################################ POS-tagger ###############################

POSState = namedtuple('POSState', 'chars words index outputs')

STOP_AFTER_SAMPLES = 10 * 1000
TEST_EVERY_SAMPLES = 2000
CHAR_EMB_COUNT = 500
WORD_EMB_COUNT = 30 * 1000
HIDDEN_SIZE = 50
LR = 0.1
BATCH_SIZE = 29


# pos = ud.DataProvider(lang='english')
pos = ud.DataProvider(lang='russian')
# pos = gikrya.DataProvider(lang='russian')
train_pos_tags = pos.train_pos_tags
dev_pos_tags = pos.dev_pos_tags

tags = tuple(pos.pos_tags) + (TAG_START, TAG_END)
tagmap = dict(zip(sorted(tags), range(len(tags))))
id2tag = dict(zip(range(len(tags)), sorted(tags)))
tags_count = len(tagmap)


def word_char_emb(str_word):
    assert isinstance(str_word, str)
    w = ' ' + str_word + ' '
    wl = len(w)
    ec = CHAR_EMB_COUNT
    emb = torch.zeros(ec)
    n = 3
    for i in range(wl - n + 1):
        emb[hash(w[i:i+n+1]) % ec] = 1
    return emb


def char_emb_ids(word: str, embeddings_count, embedding_length=3):
    w = ' ' + word + ' '
    n = len(w)
    return [hash(w[i:i + n + 1]) % embeddings_count for i in range(n - embedding_length + 1)]


class POS(nn.Module):
    def __init__(self, hidden_size, tags_count):
        super().__init__()
        self.hidden_size = hidden_size
        self.tag_emb = nn.Embedding(tags_count, hidden_size)
        self.char_emb = nn.EmbeddingBag(CHAR_EMB_COUNT, hidden_size, mode='sum')
        self.word_emb = nn.Embedding(WORD_EMB_COUNT, hidden_size)

        self.W = nn.Linear(3 * hidden_size, tags_count)

    def create_initial_states(self, inputs):
        res = []
        for input in inputs:
            words = (' ',) + input + (' ',)
            char_input = [char_emb_ids(w, CHAR_EMB_COUNT) for w in words]
            res.append(POSState(char_input, words, 1, (tagmap[TAG_START], )))
        return res

    def act(self, states, actions: List[int]):
        return [POSState(s.chars, s.words, s.index+1, s.outputs + (a, )) for s, a in zip(states, actions)]

    def forward(self, states):
        char_offsets = []
        words = []
        for s in states:
            assert s.index < len(s.chars) - 1
            words.append(s.words[s.index-1:s.index+2])
            char_offsets.append([len(c) for c in s.chars[s.index-1:s.index+2]])
        char_offsets = [wo for co in char_offsets for wo in co]
        char_offsets.insert(0, 0)
        char_offsets.pop()
        offsets = Variable(torch.cumsum(torch.LongTensor(char_offsets), 0))

        prev_tag_ids = Variable(torch.LongTensor([s.outputs[s.index-1] for s in states]))

        char_ids = list(chain(*chain(*[s.chars[s.index-1:s.index+2] for s in states])))
        X = self.char_emb.forward(Variable(torch.LongTensor(char_ids)), offsets=offsets)
        X = X.view(len(states), -1)

        X[:, :self.hidden_size] = X[:, :self.hidden_size] + self.tag_emb(prev_tag_ids)

        WX = self.word_emb.forward(Variable(torch.LongTensor([[hash(w) % WORD_EMB_COUNT for w in ws] for ws in words])))
        WX = WX.view(WX.size(0), -1)

        X = X + WX
        res = F.relu(self.W.forward(X))

        res -= res.max(1, keepdim=True)[0]
        return res.exp()


pos_model = POS(HIDDEN_SIZE, tags_count)

train_sents = [tuple(zip(*sent)) for sent in pos.train_pos_tags]
test_sents = [tuple(zip(*sent)) for sent in pos.dev_pos_tags]


def batch_generator(seq: List, batch_size):
    while True:
        indexi = torch.randperm(len(seq))
        for i in range(0, len(indexi), batch_size):
            yield [seq[k] for k in indexi[i:i+batch_size]]


optimizer = Adam(pos_model.parameters(), LR)
criterion = nn.CrossEntropyLoss()

seen_samples = 0
losses = []
for batch in batch_generator(train_sents, BATCH_SIZE):
    seen_samples += len(batch)

    loss = Variable(torch.zeros(1))

    inputs, batch_tags = zip(*batch)
    batch_tags = [list(tags) for tags in batch_tags]
    states = pos_model.create_initial_states(inputs)

    state_max_len = max([len(s.chars)-2 for s in states])
    words_seen = 0
    correct_words = 0
    example = []
    for i in range(state_max_len):
        decisions = pos_model.forward(states)
        decisions /= decisions.sum(1, keepdim=True)
        ys = [tagmap[tag.pop(0)] for tag in batch_tags]
        loss += criterion(decisions, Variable(torch.LongTensor(ys))) * len(states)
        words_seen += len(states)

        _, argmax = decisions.max(1)
        correct_words += (argmax == Variable(torch.LongTensor(ys))).long().sum().data[0]

        example.append((states[0].words[states[0].index], id2tag[ys[0]], id2tag[argmax.data[0]]))

        new_states = pos_model.act(states, ys)

        states = [s for s in new_states if s.index < len(s.chars) - 1]
        batch_tags = [tag for tag in batch_tags if tag]

    assert not states

    loss = loss / words_seen
    optimizer.zero_grad()
    loss.backward()
    losses.append(loss.data[0])

    optimizer.step()

    # print(example)
    print('{:.2f}'.format(seen_samples).ljust(8), '{:.1f}%'.format(correct_words/words_seen*100), np.mean(losses[-10:]), sep='\t')

    if (seen_samples // BATCH_SIZE) % (TEST_EVERY_SAMPLES // BATCH_SIZE) == 0:
        inputs, test_tags = zip(*test_sents)
        test_tags = [list(tags) for tags in test_tags]
        states = pos_model.create_initial_states(inputs)

        state_max_len = max([len(s.chars) - 2 for s in states])
        words_seen = 0
        correct_words = 0
        loss = 0
        example = []
        for i in range(state_max_len):
            decisions = pos_model.forward(states)
            decisions /= decisions.sum(1, keepdim=True)
            ys = [tagmap[tag.pop(0)] for tag in test_tags]
            loss += criterion(decisions, Variable(torch.LongTensor(ys))).data[0] * len(states)
            words_seen += len(states)

            _, argmax = decisions.max(1)
            correct_words += (argmax == Variable(torch.LongTensor(ys))).long().sum().data[0]

            example.append((states[0].words[states[0].index], id2tag[ys[0]], id2tag[argmax.data[0]]))

            new_states = pos_model.act(states, ys)

            states = [s for s in new_states if s.index < len(s.chars) - 1]
            test_tags = [tag for tag in test_tags if tag]
        assert not states
        loss /= words_seen
        print('Test'.ljust(8), '{:.1f}%'.format(correct_words/words_seen*100), '{:.2f}'.format(loss).ljust(8), sep='\t')
        print('Test'.ljust(8), example)

    # if seen_samples > STOP_AFTER_SAMPLES:
    #     break
