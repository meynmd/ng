#!/usr/bin/python

from __future__ import print_function
from __future__ import division
import sys
import string
import random
import numpy
import scipy.stats
import math
from collections import defaultdict
from itertools import product
from copy import *


def Train(filename, gram = 3):
    seqCounts, ngramCounts = Count(filename, gram)
    if gram == 1:
        total = sum(ngramCounts.values())
        ngram = {}
        ngram['uni'] = defaultdict(float)
        for char, count in ngramCounts.items():
            ngram['uni'][char] = float(count) / total
    else:
        ngram = FindProbabilities(seqCounts, ngramCounts)
    return ngram


def Load(filename):
    data = ''
    fp = open(filename, 'r')
    return fp.read()


def Count(filename, gram):
    preCount = defaultdict(int)

    # make set of chars
    data = Load(filename)
    data = data.replace(' ', '_')
    data = data.replace('\n', '$')
    chars = set([c for c in data])
    data = string.lower(data)
    data = (gram - 1) * '#' + data
    data = data.replace('$', '$' + (gram - 1) * '#')

    # set up ngram counts
    if gram == 1:
        ngramCount = defaultdict(int)
        for line in data.split('$'):
            line += '$'
            for char in line:
                ngramCount[char] += 1
        return None, ngramCount
    
    ngramCount = {}
    pre = [''.join(
        [x for x in p]
    ) for p in product(chars, repeat = gram - 1)]
    pre += (gram - 1) * '#'

    for i in range(1, gram):
        pre += [i * '#' + ''.join(
            [x for x in p]
        ) for p in product(chars, repeat = gram - 1 - i)]
    #chars.add('#')
    for p in pre:
        ngramCount[p] = {x : 0 for x in chars}

    # count occurrences
    for line in data.split('$'):
        for i in range(len(line) - gram - 1):
            sequence = line[i : i + gram - 1]
            nextChar = line[i + gram - 1]
            ngramCount[sequence][nextChar] += 1
            preCount[sequence] += 1

    return preCount, ngramCount


def FindProbabilities(seqCount, ngramCount, k = 5):
    # freq : occurrences of each count
    freq = [0 for x in range(
        int(max(max(cc.values()) for cc in ngramCount.values())) + 1)]
    for seq, char_count in ngramCount.items():
        for char, count in char_count.items():
            freq[count] += 1

    maxSeqCount = max(seqCount.values())

    # smooth the counts
    ngramProb = defaultdict(defaultdict)
    totals = defaultdict(float)
    a, b, _, _, _ = scipy.stats.linregress(range(len(freq)), freq)

    for seq, char_count in ngramCount.items():
        sc = seqCount[seq]
        sw = math.exp(-2. * sc / maxSeqCount)
        v = numpy.std(char_count.values())
        k = math.ceil(math.log(v + 1) * sw)

        realProb = {}
        reserved = 0.
        probSum = 0.
        totals[seq] = n = sum(char_count.values())
        for char, count in char_count.items():
            if n <= 0.:
                # no observed successors for this sequence
                ngramProb[seq][char] = p = 1. / len(char_count.keys())
                reserved += p
                continue

            # record count of every n-gram seen at least k times
            if count > k:
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
                    ngramProb[seq][char] = p = (1./sw) * (count + 1) * (ncp1 / (totals[seq] * nc))
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


def MakeFSA(ngram, order, startSymbol = '<s>', endSymbol = '</s>'):
    if order == 1:
        #fsa = 'F\n(S (F </s> 1.0))\n'
        fsa = 'F\n(S (1 <s> 1.0))\n'
        #fsa += '(S (S {0} 1.0))\n'.format(startSymbol)
        for seq, char_prob in ngram.items():
            for char, prob in char_prob.items():
                if char == '$':
                    fsa += '(1 (F {0} {1}))\n'.format(endSymbol, prob)
                else:
                    fsa += '(1 (1 {0} {1}))\n'.format(char, prob)
        return fsa

    fsa = 'F\n(S ({0} {1} 1.0))\n'.format((order - 1) * '#', startSymbol)
    for seq, char_prob in ngram.items():
        for char, prob in char_prob.items():
            if char == '$':
                fsa += '({0} (F {1} {2}))\n'.format(
                    seq, endSymbol, prob)
            else:
                fsa += '({0} ({1} {2} {3}))\n'.format(
                    seq, seq[1 :] + char, char, prob)
    return fsa



if __name__ == "__main__":
    filename = sys.argv[1]
    gram = int(sys.argv[2])
    outfile = sys.stdout
    if len(sys.argv) > 3:
        outfile = open(sys.argv[3], 'w')

    model = Train(filename, gram)

    print(MakeFSA(model, gram), file=outfile)
