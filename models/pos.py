from collections import namedtuple
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np

from data_providers.ud_pos.pos import DataProvider



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

POSState = namedtuple('POSState', 'input index outputs')

EPOCHS = 100
WORD_DIM = 1000
HIDDEN_SIZE = 30
T = 2
LR = 0.005
BATCH_SIZE = 29


pos = DataProvider(lang='russian')
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
    ec = WORD_DIM
    emb = torch.zeros(ec)
    n = 3
    for i in range(wl - n + 1):
        emb[hash(w[i:i+n+1]) % ec] = 1
    return emb


class POS(nn.Module):
    def __init__(self, word_emb_size, tags_count):
        super().__init__()
        self.W = nn.Linear(word_emb_size, tags_count)
        self.tag_emb = nn.Embedding(tags_count, tags_count)

    def create_initial_states(self, inputs):
        res = []
        for input in inputs:
            words = ('',) + input + ('',)
            tensor_input = torch.stack([word_char_emb(w) for w in words])
            res.append(POSState(tensor_input, 1, (tagmap[TAG_START], )))
        return res

    def act(self, states, actions: List[int]):
        return [POSState(s.input, s.index+1, s.outputs + (a, )) for s, a in zip(states, actions)]

    def forward(self, states):
        for state in states:
            assert state.index < len(state.input) - 1

        prev_tag_ids = Variable(torch.LongTensor([s.outputs[s.index-1] for s in states]))

        X = Variable(torch.stack([s.input[s.index] for s in states]))
        res = self.W.forward(X) + self.tag_emb.forward(prev_tag_ids)
        return res.exp()


pos_model = POS(WORD_DIM, tags_count)

X = [('hi', 'there'), ('wazzup',)]

states = pos_model.create_initial_states(X)
assert states[0].input.size() == torch.Size([4, WORD_DIM])
assert pos_model.forward(states).size() == torch.Size([2, 18])

train_sents = [tuple(zip(*sent)) for sent in pos.train_pos_tags]

def batch_generator(seq: List, batch_size):
    while True:
        indexi = torch.randperm(len(seq))
        for i in range(0, len(indexi), batch_size):
            yield [seq[k] for k in indexi[i:i+batch_size]]

seen_samples = 0
optimizer = Adam(pos_model.parameters(), LR)
criterion = nn.CrossEntropyLoss()

losses = []
for batch in batch_generator(train_sents, BATCH_SIZE):
    loss = Variable(torch.zeros(1))

    inputs, batch_tags = zip(*batch)
    batch_tags = [list(tags) for tags in batch_tags]
    states = pos_model.create_initial_states(inputs)

    state_max_len = max([len(s.input)-2 for s in states])
    words_seen = 0
    for i in range(state_max_len):
        decisions = F.softmax(pos_model.forward(states))
        ys = [tagmap[tag.pop(0)] for tag in batch_tags]
        loss += criterion(decisions, Variable(torch.LongTensor(ys))) * len(states)
        words_seen += len(states)

        new_states = pos_model.act(states, ys)

        states = [s for s in new_states if s.index < len(s.input) - 1]
        batch_tags = [tag for tag in batch_tags if tag]

    optimizer.zero_grad()
    loss = loss / words_seen
    loss.backward()
    losses.append(loss.data[0])

    optimizer.step()

    seen_samples += len(batch)
    print('{:.2f}'.format(seen_samples).ljust(8), np.mean(losses[-10:]))


