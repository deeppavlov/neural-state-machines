import spacy
from utils.syntax import *
from pprint import pprint


nlp = spacy.load('en')
doc = nlp(u'Autonomous cars shift insurance liability toward manufacturers')

train, test, dictionary, tags_dictionary, deprel_dictionary = cached('downloads/ud_pos_deprel.pickle',
                                                                     lambda: prepare_data('ud', 'english'))


def align(spacy_doc, conllu_words):
    si = 0
    ci = 0
    while ci < len(conllu_words):
        for ln in range(60):
            if spacy_doc[si:si+ln+1].text == conllu_words[ci][1]:
                for i in range(ln+1):
                    yield spacy_doc[si+i], conllu_words[ci]
                si += ln+1
                ci += 1
                break
        else:
            raise RuntimeError()

correct_count = 0
total_count = 0
for words in test:
    if len(words) == 1:
        continue

    doc = nlp(' '.join(w[1] for w in words))
    # assert len(doc) == len(words)
    for d, w in align(doc, words):
        w_id = d.i + 1
        head_id = d.head.i + 1
        if w_id == head_id:
            head_id = 0
        # print(w_id, head_id, w[4])

        correct_count += head_id == w[4]
        total_count += 1

print('accuracy:', correct_count/total_count)
print('token count:', total_count)
