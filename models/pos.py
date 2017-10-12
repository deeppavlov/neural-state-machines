from collections import namedtuple
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

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
LR = 0.5


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

    def create_initial_states(self, inputs):
        res = []
        for input in inputs:
            words = ('',) + input + ('',)
            tensor_input = torch.stack([word_char_emb(w) for w in words])
            res.append(POSState(tensor_input, 1, ()))
        return res

    def act(self, states, actions: List[int]):
        return [POSState(s.input, s.index+1, s.outputs + (a, )) for s, a in zip(states, actions)]

    def forward(self, states):
        for state in states:
            assert state.index < len(state.input) - 1

        X = Variable(torch.stack([s.input[s.index] for s in states]))
        return self.W.forward(X).exp()


pos = POS(WORD_DIM, tags_count)
states = pos.create_initial_states([('hi', 'there'), ('wazzup',)])
assert states[0].input.size() == torch.Size([4, WORD_DIM])
assert pos.forward(states).size() == torch.Size([2, 18])

