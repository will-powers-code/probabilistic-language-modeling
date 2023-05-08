#!/bin/python

from __future__ import print_function

from lm import LangModel
import random
from math import log
import numpy as np

def is_number(word):
    try:
        float(word)
        return True
    except ValueError:
        return False

def isValidWord(word, incl_eos):
    if word == "start_of_sentence": return False
    if word == '<unk>': return False
    if not incl_eos and word == "end_of_sentence": return False
    if is_number(word): return False
    return True

class Sampler:

    # temp = .5, TopP=.1
    def __init__(self, lm, temp = 1.1, TopP=.1):
        """Sampler for a given language model.

        Supports the use of temperature, i.e. how peaky we want to treat the
        distribution as. Temperature of 1 means no change, temperature <1 means
        less randomness (samples high probability words even more), and temp>1
        means more randomness (samples low prob words more than otherwise). See
        simulated annealing for what this means.
        """
        self.lm = lm
        self.rnd = random.Random()
        self.temp = temp
        self.TopP = TopP

    def sample_sentence(self, prefix = (), max_length = 100, incl_eos = True, validWordKeysFirst=None, validWordKeysRest= None  ):
        self.lm.c=0
        """Sample a random sentence (list of words) from the language model.

        Samples words till either EOS symbol is sampled or max_length is reached.
        Does not make any assumptions about the length of the context.
        """

        validWordKeysFirst = validWordKeysFirst or [wordKey for wordKey in self.lm.vocab if isValidWord(wordKey[0],False)]
        validWordKeysRest =  validWordKeysRest or [wordKey for wordKey in self.lm.vocab if isValidWord(wordKey[0],incl_eos)]

        # print("prefix",prefix)
        i = 0
        sent = ("start_of_sentence",)+prefix
        #print("sampling first:")
        word = self.sample_next(sent, False,validWordKeysFirst)
        sent+=word
        #print(word)
        #print("entering while loop")
        while i <= max_length and word != ("end_of_sentence",):
            #print("sampling next:")
            word = self.sample_next(sent,incl_eos,validWordKeysRest)
            sent+=word
            #print(word)
            i += 1
        return sent

    def sample_next(self, prev, incl_eos = True, validWordKeys=None):
        """Samples a single word from context.

        Can be useful to debug the model, for example if you have a bigram model,
        and know the probability of X-Y should be really high, you can run
        sample_next([Y]) to see how often X get generated.

        incl_eos determines whether the space of words should include EOS or not.
        """
        if validWordKeys == None:
            validWordKeys = [wordKey for wordKey in self.lm.vocab if isValidWord(wordKey[0],incl_eos)]


        #print("entering first sample loop")
        cond_prob = self.lm.cond_prob

        #collect augmented word probabilities
        wps = [[wordKey, cond_prob(wordKey, prev)**self.temp] for wordKey in validWordKeys]

        wps.sort(key=lambda wp:wp[1], reverse=True)
        #total mass
        tot = sum((wp[1] for wp in wps))
        #normalize probabilities
        for wp in wps: wp[1] = wp[1]/tot
        
        class data:pass
        data.topIndex = 0
        topP = self.TopP
        data.currP = 0
        for i in range(len(wps)):
            data.currP+=wps[i][1]
            if data.currP>=topP:
                data.topIndex = i
                break
        topWords = wps[:data.topIndex+1]

        return self.rnd.choices([x[0] for x in topWords],[ x[1] for x in topWords])[0]

if __name__ == "__main__":
    from lm import Unigram
    unigram = Unigram()
    corpus = [
        [ "sam", "i", "am" ]
    ]
    unigram.fit_corpus(corpus)
    print(unigram.model)
    sampler = Sampler(unigram)
    for i in range(10):
        print(i, ":", " ".join(str(x) for x in sampler.sample_sentence()))

