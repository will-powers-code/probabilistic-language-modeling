#!/bin/python
from math import log
from collections import defaultdict

from math import prod

from numpy import array


class LangModel:
    #Create Word Counts and *Normalize*
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""
        # Count Words in Corpus and End Of Sentance Tokens
        for s in corpus:
            self.fit_sentence(s)
        # *Normalize* Word Counts...
        self.norm()

    #Return Perplexity Score given Model and Corpus
    def perplexity(self, corpus, n):
        self.n = n
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))
    
    #Calculate Entropy of Text for Perplexity Score
    def entropy(self, corpus):
        #Num Words in Corpus
        num_words = 0.0
            # Sum of the log of the probabilities of the Sentances 
        sum_logprob = 0.0

        for s in corpus:
            # Count Num Words and EOS Tokens
            num_words += len(s) + 1
            # Add up all Log Probabilities of Sentances
            sum_logprob += self.logprob_sentence(s)
        sum_logprob
        # Returns negative of the Sum of the Log Prob divided by the corpus word count
        return -(1.0/num_words)*(sum_logprob)

    #Calculate the Log Probability of a sentance 
    def logprob_sentence(self, sentence):
        # Sum of Probabilities
        p = 0.0
        # Loop over all words in a sentance
        for i in range(len(sentence)):
            #Add the Conditional Log Probability of a word Given its previous words
            p += self.cond_logprob(sentence[i], sentence[:i])
        # Add the conditional Probability of EOS given the previous words
        p += self.cond_logprob('END_OF_SENTENCE', sentence)
        return p

    # required, update the model when a sentence is observed
    def fit_sentence(self, sentence): pass
    # optional, if there are any post-training steps (such as normalizing probabilities)
    def norm(self): pass
    # required, return the log2 of the conditional prob of word, given previous words
    def cond_logprob(self, word, previous): pass
    # required, the list of words the language model suports (including EOS)
    def vocab(self): pass

class Unigram(LangModel):

    #Init Prob Dist. (Dict) and log Backoff
    def __init__(self, backoff = 0.000001):
        #Initiate Probability Distribution as Dict
        self.models = dict()
        #Store Log_2 of Backoff Parameter
        self.lbackoff = log(backoff, 2)

    # Increment Word Count
    def inc_word(self, w):
        if w in self.models:
            self.models[w] += 1.0
        else:
            self.models[w] = 1.0
    
    #Count Words in Sentance + EndOfSentance Token
    def fit_sentence(self, sentence):
        for w in sentence:
            self.inc_word(w)
        self.inc_word('END_OF_SENTENCE')

    #converts counts to Log of Empirical Frequencies
    def norm(self):
        """Normalize and convert to log2-probs."""
        #Get Total Count of Words
        tot = 0.0
        for word in self.models:
            tot += self.models[word]
        #Get Log_2 of #Words
        ltot = log(tot, 2)

        #Convert Counts to something to be used for normalization....?
        for word in self.models:
            #Get Log of Words Counts and Subtract Log Total Count
            self.models[word] = log(self.models[word], 2) - ltot
    
    # return Normalized Word Count OR return log of Backoff
    def cond_logprob(self, word, previous):
        if word in self.models:
            return self.models[word]
        else:
            return self.lbackoff

    #Return List of Words
    def vocab(self):
        return self.models.keys()

class Ngram(LangModel):
    #Init Prob Dist. (Dict) and log Backoff
    def __init__(self, maxN=3, discount=.5, discardedCountSize=1):
        #Initiate ngram counts as Dicts
        self.models = dict()
        for i in range (1,maxN+1):
            self.models[i] = defaultdict(lambda:0.0)
        #Store Log_2 of Backoff Parameter
        self.discardedCountSize = discardedCountSize
        #Store ngram length
        self.maxN = maxN
        #store discount factor
        self.discount = discount
        #caches
        self.MLWordDict = defaultdict(lambda:None)
        self.vocabList = dict()
        aSetCounts = dict()
        self.aSetCounts = aSetCounts
        aSetBCounts = dict()
        self.aSetBCounts = aSetBCounts
        katzProbDict = defaultdict(lambda:None)
        PMandBSetDenomDict = defaultdict(lambda:1.0)
        self.katzProbDict = katzProbDict
        self.PMandBSetDenomDict = PMandBSetDenomDict


    #Create Word Counts and *Normalize*
    def fit_corpus(self, corpus):
        """Learn the language model for the whole corpus.

        The corpus consists of a list of sentences."""  
        #******* Local Vars *******
        models = self.models
        MLWordDict = self.MLWordDict
        model1 = self.models[1]
        #******* Local Vars *******

        #Replace Uncommon Words
        tCorpus = self.trainingCorpus(corpus)

        # Count n-grams in Corpus
        nGramRange = range(1,self.maxN+1)
        m = [models[n] for n in nGramRange]
        for sentence in tCorpus:
            senLen = len(sentence)
            for n in nGramRange:
                modelN = m[n-1]
                for i in range(senLen-n+1):
                    modelN[sentence[i:i+n]] += 1.0

        #print(model1[("<unk>",)])

        # memoize Total Words
        totalWords = sum(model1.values())
        self.totalWords = totalWords

        # memoize Vocab
        vocabList = self.vocabList
        for n in range(1, self.maxN+1):
            vocabList[n] = list(models[n].keys())
        self.vocab = vocabList[1]

        # memoize num vocab
        self.numVocab = len(vocabList[1])

        #print("Memoize ML Estimations")
        # Memoize ML Estimations
        for wordKey in vocabList[1]:
            MLWordDict[wordKey] = model1[wordKey]/totalWords

        
        #******* Local Vars *******

        #******* Local Vars *******

        ###############################################
        ##### Memoize PMandBSetDenom and KatzProb #####
        ###############################################
        
        for n in range(1,self.maxN):
            #print("Memoize ABSet size,", n)
            self.memoizeABSet(n)
            #print("Memoize PMandBSetDenom size,", n)
            self.memoizePMandBSetDenomSize(n)
            self.aSetCounts[n].clear()
            self.aSetBCounts[n].clear()
            #print("Memoize known Katz size n,", n+1)
            self.memoizekatzProb_Known(n+1)
        
    #replace uncommon words or lowercase + add EOS and SOS Tokens
    def trainingCorpus(self, corpus):
        addBoundaryTokens = self.addBoundaryTokens
        freq = defaultdict(lambda:0)
        tCorpus = []
        
        #get frequency
        for s in corpus:
            for w in s:
                freq[w.lower()] += 1.0
        
        #replace low frequency words and add EOS and SOS Tokens
        discardedCountSize = self.discardedCountSize
        for si in range(len(corpus)):
            s = [*corpus[si]]
            for wi in range(len(s)):
                    s[wi] = "<unk>" if freq[s[wi]] <= discardedCountSize else s[wi].lower()
            addBoundaryTokens(s)
            tCorpus.append(tuple(s))
        return tCorpus

    def addBoundaryTokens(self, sentence):
        if sentence[len(sentence)-1] != 'end_of_sentence':
            sentence.append('end_of_sentence')
        if sentence[0] != 'start_of_sentence':
            sentence.insert(0,"start_of_sentence")

    def changeModel(self, n):
        if n > self.maxN or n < 2:
            raise Exception("Invalid ngram length")
        self.n = n


    #Return Perplexity Score given Model and Corpus
    def perplexity(self, corpus, n):
        self.changeModel(n)
        """Computes the perplexity of the corpus by the model.

        Assumes the model uses an EOS symbol at the end of each sentence.
        """
        return pow(2.0, self.entropy(corpus))

    def entropy(self, corpus):
        #******* Local Vars *******
        modelOne = self.models[1]    
        addBoundaryTokens = self.addBoundaryTokens
        katzProbDict = self.katzProbDict
        n = self.n
        katzProb = self.katzProb
        #******* Local Vars *******

        #Num Tokens in Corpus
        num_words = 0

        tCorpus = []
        unkCount = 0
        #Add EOS, SOS, and <UNK> Tokens to Corpus
        for si in range(len(corpus)): 
            sentence = [*corpus[si]]
            for wi in range(len(sentence)):
                num_words += 1
                word = sentence[wi].lower()
                if modelOne[(word,)] == 0: 
                    sentence[wi] = "<unk>"
                    unkCount+=1
                else: sentence[wi] = word
            addBoundaryTokens(sentence)
            tCorpus.append(tuple(sentence))
        #print(unkCount)

        # Sum of the log of the probabilities of the Sentances 
        log_sum = sum(sum(log(katzProbDict[sentence[:i+1][-n:]] or katzProb(sentence[i:i+1], sentence[:i][-n+1:])) for i in range(1,len(sentence))) for sentence in tCorpus)

        # Returns negative of the Sum of the Log Prob divided by the corpus word count
        return -(1.0/num_words)*log_sum

    def prob_sentence(self, sentence):
        katzProbDict = self.katzProbDict
        n = self.n
        katzProb = self.katzProb
        return prod(katzProbDict[sentence[:i+1][-n:]] or katzProb(sentence[i:i+1], sentence[:i][-n+1:]) for i in range(1,len(sentence)))

    #Used for Generator
    def cond_prob(self, word, previousKey,):
        return self.katzProbDict[previousKey[-self.n+1:]+word] or self.katzProb(word, previousKey[-self.n+1:])
    
    def katzProb(self, wordKey, previousKey):
        PMandBSetDenom = self.PMandBSetDenomDict[previousKey]
        fullKey = previousKey+wordKey

        if len(previousKey) == 1:
            p = PMandBSetDenom * self.MLWordDict[wordKey]
            self.katzProbDict[fullKey] = p
            return p
        else:
            p = PMandBSetDenom * (self.katzProbDict[fullKey[1:]] or self.katzProb(wordKey, previousKey[1:]))
            self.katzProbDict[fullKey] = p
            return p

    def memoizekatzProb_Known(self, n):
        models = self.models
        modelN = models[n]
        vocabN = self.vocabList[n]
        modelNM1 = models[n-1]
        d = self.discount

        kpDict = self.katzProbDict
        for vN in vocabN:
            kpDict[vN] = (modelN[vN]-d)/modelNM1[vN[:-1]]
                
    def memoizePMandBSetDenomSize(self, n):
        #**** FUNC. SCOPE OPTIMIZE VARS ****
        models = self.models
        modelN = models[n]
        aSetCountN = self.aSetCounts[n]
        aSetBCountN = self.aSetBCounts[n]
        vocabN = self.vocabList[n]
        PMandBSetDenomDict = self.PMandBSetDenomDict
        #***********************************
        for previousKey in vocabN:
            #### MPM ##################
            missingProbMass = 1 - (aSetCountN[previousKey]/modelN[previousKey])
            ############################
            PMandBSetDenomDict[previousKey] = missingProbMass/(1 - aSetBCountN[previousKey])          

    def memoizeABSet(self,n):
        aSetCounts = self.aSetCounts
        aSetBCounts = self.aSetBCounts
        
        aSetCountsN = defaultdict(lambda:0)
        aSetCounts[n] = aSetCountsN
        aSetBCountsN = defaultdict(lambda:0)
        aSetBCounts[n] = aSetBCountsN

        modelNp1 = self.models[n+1]
        MLWordDict = self.MLWordDict
        katzProbDict = self.katzProbDict
        vocabNp1 = self.vocabList[n+1]
        d=self.discount
        
        for vNp1 in vocabNp1:
            prev = vNp1[:-1]
            aSetCountsN[prev] += modelNp1[vNp1] - d
            aSetBCountsN[prev] += MLWordDict[vNp1[1:]] if n == 1 else katzProbDict[vNp1[1:]]
