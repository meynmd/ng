#!/usr/bin/python

from __future__ import print_function
from __future__ import division
import sys
import string
import random
import scipy.stats
import math
from collections import defaultdict
from itertools import product
from copy import *


def Load(filename):
    data = ''
    fp = open(filename, 'r')
    return fp.read()


def Train(filename, gram = 3):
    # make set of chars
    data = Load(filename)
    data = data.replace(' ', '_')
    data = data.replace('\n', '$')
    chars = set([c for c in data])
    data = string.lower(data)
    data = (gram - 1) * '#' + data

    # set up ngram dictionary with no nonzero values
    ngram = {}
    count = 0
    pre = [''.join([
        x for x in p]) for p in product(chars, repeat = gram - 1
    )]
    pre += (gram - 1) * '#'
    for i in range(1, gram):
        pre += [i * '#' + ''.join(
            [x for x in p]
        ) for p in product(chars, repeat = gram - 1 - i)]

    for p in pre:
        ngram[p] = {x : 0 for x in chars}
        count += len(chars)

    # count occurrences
    for i in range(len(data) - gram - 1):
        sequence = data[i : i + gram - 1]
        nextChar = data[i + gram - 1]
        ngram[sequence][nextChar] += 1

    smoothCount, totals = Smooth(ngram)

    # calculate probabilities
    ngram = defaultdict(defaultdict)
    for seq, char_count in smoothCount.items():
        #n = sum(char_count.values())
        for char, count in char_count.items():
            ngram[seq][char] = float(count) / totals[seq]
            if ngram[seq][char] > 1.:
                print(seq, char, count, totals[seq], file=sys.stderr)

    for p in pre:
        for c in chars:
            if c not in ngram[p]:
                print(p, c, file=sys.stderr)

    return ngram


def Smooth(ngramCount):
    k = 1
    smoothedModel = {}

    # freq : occurrences of each count
    freq = [0 for x in range(
        int(max(max(cc.values()) for cc in ngramCount.values())) + 1)]
    for seq, char_count in ngramCount.items():
        for char, count in char_count.items():
            freq[count] += 1

    a, b, _, _, _ = scipy.stats.linregress(range(len(freq)), freq)

    # smooth the counts
    smoothCount = defaultdict(defaultdict)
    totals = defaultdict(float)
    for seq, char_count in ngramCount.items():
        for char, count in char_count.items():
            sc = 0
            if count > 0:
                sc = ngramCount[seq][char]
            else:
                nc = freq[count]
                ncp1 = freq[count + 1]
                if nc == 0 or ncp1 == 0:
                    log_nc = b + a * math.log(count)
                    sc = log_nc
                else:
                    nk = freq[1]
                    nkp1 = freq[k + 1]
                    sc = (
                        (count + 1) * (float(ncp1) / nc) -
                        count * (k + 1) * nkp1 / nk
                        ) / (1 - (k + 1) * float(nkp1) / nk)

            smoothCount[seq][char] = sc
            totals[seq] += sc

    return smoothCount, totals


def MakeFSA(ngram, order, startSymbol = '<s>'):
    fsa = 'F\n(S (S *e* 1.0))\n'
    for seq, char_prob in ngram.items():
        for char, prob in char_prob.items():
            # if all chars are start symbol
            if seq[-1] == '#':
                fsa += '(S ({0} {1} {2}))\n'.format(
                    seq[1 :] + char, startSymbol, prob
                )
            else:
                if char == '$':
                    fsa += '({0} (F {1} {2}))\n'.format(
                        seq, '</s>', prob
                    )
                else:
                    fsa += '({0} ({1} {2} {3}))\n'.format(
                        seq, seq[1 :] + char, char, prob
                    )
    return fsa



if __name__ == "__main__":
    filename = sys.argv[1]
    order = int(sys.argv[2])
    outfile = sys.stdout
    if len(sys.argv) > 3:
        outfile = open(sys.argv[3], 'w')

    model = Train(filename, order)

    print(MakeFSA(model, order), file=outfile)
