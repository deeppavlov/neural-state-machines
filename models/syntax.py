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
    sent = []
    for w in s:
        sent.append((w.id, w.form, w.head))
    train.append(sent)

test = []
for s in conllu.dev:
    sent = []
    for w in s:
        sent.append((w.id, w.form, w.head))
    test.append(sent)

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
        sentences = [[WORD_EMPTY, WORD_ROOT] + sent for sent in sentences]

        embs, word_indexes = self._batch_char_emb(sentences)
        for i, ws in enumerate(sentences):
            buffer = embs[word_indexes[i]:word_indexes[i + 1]]

            state = SyntaxState(ws, 0, {}, [(-2, buffer[0]), (-1, buffer[0]), (0, buffer[1])], buffer[2:])
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
                    child = s.stack[-1]
                    head = s.stack[-2]

                elif a == 2:  # left-arc
                    child = s.stack[-2]
                    head = s.stack[-1]

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


tbsp = TBSyntaxParser()

data = [[w[1] for w in s] for s in train]

batch = data[:3]

states = tbsp.create_initial_states(batch)
