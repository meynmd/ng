#!/usr/bin/python

import sys
import random
import string
from collections import defaultdict
from collections import Counter
from itertools import product


def Load(filename):
    data = ''
    fp = open(filename, 'r')
    return fp.read()


def Train(filename, order = 3, lid = 0.00001):
    # make set of chars
    data = Load(filename)
    data = data.replace(' ', '_')
    data = data.replace('\n', '$')
    chars = set([c for c in data])# if c != '\n'])
    data = string.lower(data)
    data = (order - 1) * '#' + data

    # set up ngram dictionary with no nonzero values
    ngram = {}
    count = 0
    pre = [''.join([x for x in p]) for p in product(chars, repeat = order - 1)]
    pre += (order - 1) * '#'
    for i in range(1, order):
        pre += [i * '#' +
            ''.join([x for x in p]) for p in product(chars, repeat = order - 1 - i)]
    
    for p in pre:
        ngram[p] = {x : lid for x in chars}
        count += len(chars)

    # count occurrences
    for i in range(len(data) - order - 1):
        sequence = data[i : i + order - 1]
        nextChar = data[i + order - 1]
        ngram[sequence][nextChar] += 1
    
    # calculate probabilities
    for seq, char_count in ngram.items():
         n = sum(char_count.values()) + count * lid
         for char, count in char_count.items():
             ngram[seq][char] = float(count) / n

    return ngram

    
def ProbSmooth(ngram):
    smoothedModel = {}
    freq = defaultdict(int)
    for seq, char_count in ngram:
        for char, count in char_count:
            freq[count] += 1

    for seq, char_count in ngram:
        smoothedModel[seq] = {}
        for char, count in char_count:
            n = sum(char_count.values())
            smoothedModel[seq][char] = 1 / n * (count + 1) * (freq[count + 1] / freq[count])


def MakeFSA(ngram, order, startSymbol = '<s>'):
    fsa = 'F\n(S (S *e* 1.0))'
    for seq, char_prob in ngram.items():
        for char, prob in char_prob.items():
            # if all chars are start symbol
            if seq[-1] == '#':
                fsa += '(S ({0} {1} {2}))\n'.format(
                    seq[1 :] + char, startSymbol, prob
                )
            else:
                if char == '$':
                    fsa += '({0} (F {1} {2}))'.format(
                        seq, '</s>', prob
                    )
                else:
                    fsa += '({0} ({1} {2} {3}))\n'.format(
                        seq, seq[1 :] + char, char, prob
                    )
    return fsa
   
    

if __name__ == "__main__":
    order = int(sys.argv[2])
    filename = sys.argv[1]
    model = Train(filename, order)

    print MakeFSA(model, order)
