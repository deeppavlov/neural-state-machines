import os
import random
import pickle
from collections import namedtuple, defaultdict
from copy import deepcopy
from itertools import chain
from time import time
from typing import List, Iterator, Dict

import numpy as np
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn.functional as F

from utils.syntax import *

STOP_AFTER_SAMPLES = 20000 * 1000
TEST_EVERY_SAMPLES = 5000
CHAR_EMB_COUNT = 5000
WORD_EMB_COUNT = 100 * 1000
TAG_EMB_COUNT = 100
HIDDEN_SIZE = 200

WORD_DIM = 20
CHAR_DIM = 20
TAG_DIM = 20
OUT_DIM = 3

USE_DYNAMIC_ORACLE = True

INPUT_DIM = (WORD_DIM + CHAR_DIM + TAG_DIM) * (3 + 3)

# TOP_FROM_STACK = 3
# TOP_FROM_BUFFER = 3

LR = 0.05
BATCH_SIZE = 256

PUNISH = 10

SyntaxState = namedtuple('SyntaxState', 'words tags embeddings arcs stack buffer')


def char_emb_ids(word: str, embeddings_count, embedding_length=3):
    w = ' ' + word + ' '
    n = len(w)
    return [hash(w[i:i + n + 1]) % embeddings_count for i in range(n - embedding_length + 1)]


class TBSyntaxParser(nn.Module):
    def __init__(self, words_dict: Dict[str, int], tags_dict: Dict[str, int]):
        super().__init__()
        self.words_dict = deepcopy(words_dict)
        self.words_reverse_dict = {v: k for k, v in self.words_dict.items()}
        self.words_count = len(self.words_dict)

        self.tags_dict = deepcopy(tags_dict)
        self.tags_reverse_dict = {v: k for k, v in self.tags_dict.items()}
        self.tags_count = len(self.tags_dict)

        self.word_emb = nn.Embedding(self.words_count, WORD_DIM)
        self.word_emb.weight.data.normal_(0, np.sqrt(1 / INPUT_DIM))

        self.char_emb = nn.EmbeddingBag(CHAR_EMB_COUNT, CHAR_DIM, mode='sum')
        self.char_emb.weight.data.normal_(0, np.sqrt(1/INPUT_DIM))

        self.tag_emb = nn.Embedding(TAG_EMB_COUNT, TAG_DIM)
        self.tag_emb.weight.data.normal_(0, np.sqrt(1/INPUT_DIM))

        self.input2hidden = nn.Linear(INPUT_DIM, HIDDEN_SIZE)
        self.hidden2output = nn.Linear(HIDDEN_SIZE, 3)

        self.set_device = None
        self.cpu()

    def cuda(self, device_id=None):
        super().cuda(device_id)
        self.set_device = lambda x: x.cuda(device_id)

    def cpu(self):
        super().cpu()
        self.set_device = lambda x: x.cpu()

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
                                [b + 1 for b in range(len(ws) - 1)])
            states.append(state)
        return states

    def act(self, states: List[SyntaxState], actions: List[int]):
        new_states = []
        for s, a in zip(states, actions):
            if a == 0:  # shift
                if len(s.buffer) > 3:
                    ns = SyntaxState(s.words,
                                     s.tags,
                                     s.embeddings,
                                     s.arcs,
                                     s.stack + [s.buffer[0]],
                                     s.buffer[1:])
                else:
                    raise RuntimeError('illegal move')
            else:
                if len(s.stack) <= 3:
                    raise RuntimeError('illegal move')
                elif a == 1:  # l-arc
                    if len(s.buffer) > 3:
                        head = s.buffer[0]
                    else:
                        raise RuntimeError('illegal move')
                elif a == 2:  # r-arc
                    head = s.stack[-2]
                else:
                    raise RuntimeError('unknown action')

                new_arcs = dict(s.arcs)
                new_arcs[s.stack[-1]] = head
                ns = SyntaxState(s.words,
                                 s.tags,
                                 s.embeddings,
                                 new_arcs,
                                 s.stack[:-1],
                                 s.buffer)

            new_states.append(ns)
        return new_states

    @staticmethod
    def terminated(states: List[SyntaxState]):
        return [3 == len(s.buffer) and len(s.stack) == 3 for s in states]

    def get_legal_actions(self, states: List[SyntaxState]):
        stack_len = self.set_device(torch.LongTensor([len(s.stack) for s in states]))
        buff_len = self.set_device(torch.LongTensor([len(s.buffer) for s in states]))

        l0 = buff_len > 3
        l2 = stack_len > 3
        l1 = l0 * l2
        return Variable(torch.stack([l0, l1, l2], dim=1).float())

    def forward(self, states: List[SyntaxState]):
        stack_indexes = self.set_device(torch.LongTensor([s.stack[-3:] for s in states]))
        buffer_indexes = self.set_device(torch.LongTensor([s.buffer[:3] for s in states]))
        indexes = torch.cat([stack_indexes, buffer_indexes], dim=1)

        X = torch.stack([s.embeddings[indexes[i]].view(-1) for i, s in enumerate(states)])

        hid = F.relu(self.input2hidden(X))
        out = self.hidden2output(hid)

        res = out
        return res, self.get_legal_actions(states)


def get_errors(stack: List[int], buffer: List[int], heads: Dict[int, int], punishment: int = PUNISH):
    if len(stack) < 2:
        return [0, punishment, punishment]
    if len(buffer) < 1:
        return [punishment, punishment, 0]

    l_err = len([h for h in [stack[-2]] + buffer[1:] if heads.get(stack[-1], -11) == h]) + \
            len([c for c in buffer if heads[c] == stack[-1]])

    r_err = len([i for i in buffer if heads.get(stack[-1], -11) == i or heads[i] == stack[-1]])

    s_err = len([c for c in stack if heads.get(c, -11) == buffer[0]]) + \
            len([h for h in stack[:-1] if heads[buffer[0]] == h])

    return [s_err, l_err, r_err]


##################################   TRAINING   ###################################

train, test, dictionary, tags_dictionary, deprel_dictionary = cached('downloads/ud_pos_deprel.pickle',
                                                                     lambda: prepare_data('ud', 'english'))

parser = TBSyntaxParser(dictionary, tags_dictionary)

device_id = int(os.getenv('PYTORCH_GPU_ID', -1))
if device_id >= 0:
    parser.cuda(device_id)

optimizer = Adam(parser.parameters(), LR, betas=(0.9, 0.9))

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
    tag_ids = [list(ws) for ws in tag_ids]
    tags = [list(ws) for ws in tags]

    states = parser.create_initial_states(sents, tags)

    correct_actions = 0
    total_actions = 0

    correct_heads = 0
    correct_rels = 0
    total_heads = 0

    example = []
    while states:
        decisions, legal_actions = parser.forward(states)
        # decisions /= decisions + 1

        errors = [get_errors(s.stack[2:], s.buffer[:-3], h) for s, h in zip(states, heads)]
        errors = torch.FloatTensor(errors)
        rights = Variable(parser.set_device((errors - errors.min(1, keepdim=True)[0] == 0).float()))

        local_loss = criterion(decisions, rights)

        loss += local_loss * len(states)
        total_actions += len(states)



        if USE_DYNAMIC_ORACLE:
            _, parser_next_action = ((decisions - decisions.min(1, keepdim=True)[0] + 1) * legal_actions).max(1)

        else:
            # subtract [0.1 0 0] so we prefer other action than shift if possible
            r = rights - Variable(parser.set_device(torch.FloatTensor([[0.1, 0, 0]])))

            _, parser_next_action = r.max(1)

        states = parser.act(states, parser_next_action.data.tolist())

        terminated = TBSyntaxParser.terminated(states)

        for i in reversed(range(len(states))):
            if terminated[i]:
                total_heads += len(heads[i])
                for w in range(len(heads[i])):
                    if states[i].arcs[w + 1] == heads[i][w + 1]:
                        correct_heads += 1
                        # if states[i].arc_labels[w + 1] == deprels[i][w + 1]:
                        #     correct_rels += 1
                states.pop(i)
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
        tags = [list(ws) for ws in tags]

        states = parser.create_initial_states(sents,tags)

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

            terminated = TBSyntaxParser.terminated(states)

            for i in reversed(range(len(terminated))):
                if terminated[i]:
                    total_heads += len(heads[i])
                    for w in range(len(heads[i])):
                        if states[i].arcs[w + 1] == heads[i][w]:
                            correct_heads += 1
                            # if states[i].arc_labels[w + 1] == deprels[i][w]:
                            #     correct_rels += 1
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
