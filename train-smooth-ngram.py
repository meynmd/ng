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


def Train(filename, gram = 3):
    ngramCounts = Count(filename, gram)
    ngram = FindProbabilities(ngramCounts, 4)
    return ngram


def Load(filename):
    data = ''
    fp = open(filename, 'r')
    return fp.read()


def Count(filename, gram):
    # make set of chars
    data = Load(filename)
    data = data.replace(' ', '_')
    data = data.replace('\n', '$')
    chars = set([c for c in data])
    data = string.lower(data)
    data = (gram - 1) * '#' + data

    # set up ngram dictionary with no nonzero values
    ngramCount = {}
    pre = [''.join(
        [x for x in p]
    ) for p in product(chars, repeat = gram - 1)]
    pre += (gram - 1) * '#'

    for i in range(1, gram):
        pre += [i * '#' + ''.join(
            [x for x in p]
        ) for p in product(chars, repeat = gram - 1 - i)]
    for p in pre:
        ngramCount[p] = {x : 1 for x in chars}

    # count occurrences
    for i in range(len(data) - gram - 1):
        sequence = data[i : i + gram - 1]
        nextChar = data[i + gram - 1]
        ngramCount[sequence][nextChar] += 1

    return ngramCount


def FindProbabilities(ngramCount, k = 2):
    # freq : occurrences of each count
    freq = [0 for x in range(
        int(max(max(cc.values()) for cc in ngramCount.values())) + 1)]
    for seq, char_count in ngramCount.items():
        for char, count in char_count.items():
            freq[count] += 1

    # smooth the counts
    ngramProb = defaultdict(defaultdict)
    totals = defaultdict(float)
    a, b, _, _, _ = scipy.stats.linregress(range(len(freq)), freq)

    for seq, char_count in ngramCount.items():
        totals[seq] = sum(char_count.values())
        realProb = {}
        reserved = 0.
        probSum = 0.
        for char, count in char_count.items():
            # record count of every n-gram seen at least k times
            if count >= k:
                realProb[char] = p = ngramCount[seq][char] / totals[seq]
                probSum += p
            else:
                # have to smooth
                nc = freq[count]        # number seen this many times
                ncp1 = freq[count + 1]  # number seen one extra time

                # N_c+1 may be zero
                if nc == 0 or ncp1 == 0:
                    # Katz smoothing
                    ngramProb[seq][char] = p = (b + a * math.log(count)) / totals[seq]
                    reserved += p
                else:
                    # GT smoothing
                    ngramProb[seq][char] = p = (count + 1) * (ncp1 / (totals[seq] * nc))
                    reserved += p

        if probSum > 0.:
            # there is at least one observed ngram for this sequence
            w = (1. - reserved) / probSum
            for char in char_count.keys():
                if char not in ngramProb[seq].keys():
                    ngramProb[seq][char] = w * realProb[char]
        else:
            w = 1. / sum(x for x in ngramProb[seq].values())
            for key, val in ngramProb[seq].items():
                ngramProb[seq][key] = w * val

    return ngramProb


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
    gram = int(sys.argv[2])
    outfile = sys.stdout
    if len(sys.argv) > 3:
        outfile = open(sys.argv[3], 'w')

    model = Train(filename, gram)

    print(MakeFSA(model, gram), file=outfile)
