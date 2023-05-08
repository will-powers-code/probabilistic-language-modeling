#!/bin/python

from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
from sklearn.utils import shuffle

from generator import isValidWord


#Reads Texts & Tokenizes Sentances as Documents by Counts 
def read_texts(tarfname, dname):
    """Read the data from the homework data file.

    Given the location of the data archive file and the name of the
    dataset (one of brown, reuters, or gutenberg), this returns a
    data object containing train, test, and dev data. Each is a list
    of sentences, where each sentence is a sequence of tokens.
    """
    #Get Tar File
    import tarfile
    tar = tarfile.open(tarfname, "r:gz", errors = 'replace')
    #Go Through Files in Tar and Extract Text
    for member in tar.getmembers():
        #Extract Train File
        if dname in member.name and ('train.txt') in member.name:
            # Print Train File Name
                # print('\ttrain: %s'%(member.name))
            # Extract Train Text from Corpus
            train_txt = str(tar.extractfile(member).read(), errors='replace')
        #Extract Test File
        elif dname in member.name and ('test.txt') in member.name:
            # Print Test File Name
                # print('\ttest: %s'%(member.name))
            # Extract Test Text from Corpus
            test_txt = str(tar.extractfile(member).read(), errors='replace')
        #Extract Dev File
        elif dname in member.name and ('dev.txt') in member.name:
            # Print Dev File Name
                # print('\tdev: %s'%(member.name))
            # Extract Dev Text from Corpus
            dev_txt = str(tar.extractfile(member).read(), errors='replace')

    #Create Count Vectorizer
    from sklearn.feature_extraction.text import CountVectorizer
    count_vect = CountVectorizer()
    #Learning Vocab Dict From Corpus
    count_vect.fit(train_txt.split("\n"))
    #Create func. to split sentances into tokens
    tokenizer = count_vect.build_tokenizer()
    #Create Empty object
    class Data: pass
    data = Data()
    #Collect list of tokenized sentances
    data.train = []
    for s in train_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.train.append(toks)
    #Collect list of tokenized sentances
    data.test = []
    for s in test_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.test.append(toks)
    #Collect list of tokenized sentances
    data.dev = []
    for s in dev_txt.split("\n"):
        toks = tokenizer(s)
        if len(toks) > 0:
            data.dev.append(toks)
    return data

# Generate and Train Language Model
def learn_LM(data, maxNGram=3, discount=.5, verbose=True, sampling=False):
    """Learns a unigram model from data.train.

    It also evaluates the model on data.dev and data.test, along with generating
    some sample sentences from the model.
    """
    from lm import Ngram
    #Create Language Model
    AppliedLM = Ngram(maxNGram, discount, 1)
    #train Language Model
    AppliedLM.fit_corpus(data.train)
    #Prints Perplexities and Sample Sentances from LM
    if verbose:
        #print corpus filename
        print("-----------------------")
        if not sampling: print("vocab:", len(AppliedLM.vocab))
        if not sampling: print("Num Train Words:", sum([len(x) for x in data.train]))
        if not sampling: print("Train Sentances:", len(data.train))
        if not sampling: print("Dev Sentances:", len(data.dev))
        if not sampling: print("Test Sentances:", len(data.test))
        from generator import Sampler
        AppliedLM.changeModel(maxNGram)
        sampler = Sampler(AppliedLM)
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        SS1 = sampler.sample_sentence()
        print("LM sample 1: ", " ".join(SS1[1:-1]))
        #print("*Prob* = ",AppliedLM.prob_sentence(SS1))
        #print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        SS2 = sampler.sample_sentence(("the","speech",))
        print("LM sample 2: ", " ".join(SS2[1:-1]))
        #print("*Prob* = ",AppliedLM.prob_sentence(SS2))

        # evaluate on train, test, and dev
        print("train:", AppliedLM.perplexity(data.train,maxNGram))

        print("dev  :", AppliedLM.perplexity(data.dev,maxNGram))
        print("test :", AppliedLM.perplexity(data.test,maxNGram))
        print("-----------------------")
    return AppliedLM

def optmizieSampler(maxNGram=3, discount=.5):
    #list of corpus names
    dnames = ["brown", "reuters", "gutenberg"]
    # Learn the models for each of the domains, and evaluate it
    datas = getDatas(dnames)
    for i in range(len(datas)):
        data = datas[i]
        from lm import Ngram
        #Create Language Model
        AppliedLM = Ngram(maxNGram, discount, 1)
        #train Language Model
        AppliedLM.fit_corpus(data.train)
        from generator import Sampler
        AppliedLM.changeModel(maxNGram)
        print(dnames[i])
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
        temps = (1.1, 1.2)
        topPs = (.01, .05, .1)
        avgs = [0]*(len(temps)*len(topPs))
        validWordKeys = [wordKey for wordKey in AppliedLM.vocab if isValidWord(wordKey[0],False)]
        for tempI in range(len(temps)):
            for topPI in range(len(topPs)):
                temp = temps[tempI]
                topP = topPs[topPI]
                print (temp,topP)
                sampler = Sampler(AppliedLM,temp,topP)
                sum = 0
                for i in range(3):
                    SS2 = sampler.sample_sentence((),5,False,validWordKeys,validWordKeys)
                    sum += AppliedLM.prob_sentence(SS2)
                avgs[tempI*len(topPs)+topPI] += (sum/3)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
    maxavg = max(avgs)
    index = np.argmax(avgs)
    print(f"Max Prob = {maxavg} with {topPs[index%len(topPs)]} topP and {temps[index//len(topPs)]} temp")


def optmizieAdapter(maxNGram=3, discount=.9):
    #list of corpus names
    dnames = ["brown", "reuters", "gutenberg"]
    # Learn the models for each of the domains, and evaluate it
    datas = getDatas(dnames)
    from lm import Ngram
    numerators = list(range(0,11,2))
    percents = list(map(lambda x:x/10,numerators))
    resultDict = dict()
    for m in range(len(datas)):
        results = [0]*len(numerators)
        for c in range(len(datas)):
            if c == m: 
                continue
            model = datas[m]
            corpus = datas[c]
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
            for i in range(len(numerators)):
                n = numerators[i]
                AppliedLM = Ngram(maxNGram, discount, 1)
                AppliedLM.fit_corpus(model.train+corpus.train[:(len(corpus.train)//10)*n])
                perp = AppliedLM.perplexity(corpus.dev,maxNGram)
                if n == 10:
                    print(f"model {dnames[m]} + {dnames[c]} corpus: {perp}")
                
                results[i] += perp
        resultDict[m] = results
        minPerp = min(results)
        minPerpI = np.argmin(results)
        winningPercent = percents[minPerpI]
        print(f"the minimizing percent is {winningPercent} with {minPerp}")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~")
    plt.ioff()

    plt.figure()
    plt.xlabel('Percent of Outside Corpus Train set Used vs Perplexity')
    plt.ylabel('Sum of Outside Perplexities')
    plt.title(f'Percent of Outside Corpus Used vs Dev Set Perplexity')

    colors = {3:['m-','r-','b-'],
    4:['m--','r--','b--'],
    5:['m.-','r.-','b.-'],}

    for m in range(len(datas)):
        plt.plot(percents, resultDict[m], colors[3][m], label=f"{dnames[m]}")

    plt.legend(loc='upper right')
    plt.show()
            

         

#Prints Table (Poorly Used)
def print_table(table, row_names, col_names):
    """Pretty prints the table given the table, and row and col names.

    If a latex_file is provided (and tabulate is installed), it also writes a
    file containing the LaTeX source of the table (which you can \input into your report)
    """
    try:
        from tabulate import tabulate
        row_format ="{:>15} " * (len(col_names) + 1)
        rows = list(map(lambda rt: [rt[0]] + rt[1], zip(row_names,table.tolist())))

        print(tabulate(rows, headers = [""] + col_names))
        # if latex_file is not None:
        #     latex_str = tabulate(rows, headers = [""] + col_names, tablefmt="latex")
        #     with open(latex_file, 'w') as f:
        #         f.write(latex_str)
        #         f.close()
    except ImportError as e:
        for row_name, row in zip(row_names, table):
            print(row_format.format(row_name, *row))

def plotPerplexityByCorpus(datas, models, dnames):
    
    colors = ['m-','r-','b--']

    def getData(current_corpus):
        batch_size = int(len(datas[current_corpus].train)/float(10))
        trainset = shuffle(datas[current_corpus].train)
        trainingCorpus = trainset[:batch_size*10]
        chunks = [ trainingCorpus[:i+batch_size] for i in range(0, len(trainingCorpus), batch_size)]

        perplexities = []
        chunks_lens = []
        for chunk in chunks:
            chunks_lens.append( len(chunk))
            class data: pass
            data.train = chunk
            model = learn_LM(data, 1, .5, False)
            perplexities.append(model.perplexity(datas[current_corpus].test,1)) 
        return (perplexities,chunks_lens)

    plt.xlabel('Training data size')
    plt.ylabel('Perplexity')
    plt.title('Training Data Size vs Test Set Perplexity')

    for i in range(3):
        (perplexities,chunks_lens) = getData(i)
        plt.plot(chunks_lens, perplexities, colors[i], label=dnames[i])
    plt.legend(loc='upper right')
    plt.show()

def getDatas(dnames):
    return [read_texts("datum/corpora.tar.gz", dname) for dname in dnames]

def trainModels(nGram, discount,verbose, sampling=False):
    #list of corpus names
    dnames = ["brown", "reuters", "gutenberg"]
    #corpus data
    datas = []
    #trained models by corpus
    models = []
    # Learn the models for each of the domains, and evaluate it
    datas = getDatas(dnames)
    
    for i in range(len(datas)):
        data = datas[i]
        print(dnames[i])
        models += [learn_LM(data, nGram, discount, verbose, sampling)]
        
    return(dnames, datas,models)

def perplexityMatrices(dnames, models, datas, n,dev=False):
    # n = number of corpuses/models
    numDocs = len(dnames)
    # Create 3x3 matrices of 0
    #Dev
    perp_dev = np.zeros((numDocs,numDocs))
    #Test
    perp_test = np.zeros((numDocs,numDocs))
    #Train
    perp_train = np.zeros((numDocs,numDocs))
    for i in range(numDocs):
        for j in range(numDocs):
            #print(i,j,"perplexity",)
            #populate matrices with perpexities of model i on data j
            perp_dev[i][j] = models[i].perplexity(datas[j].dev,n)
            if not dev: perp_test[i][j] = models[i].perplexity(datas[j].test,n)
            if not dev: perp_train[i][j] = models[i].perplexity(datas[j].train,n)
    return perp_train, perp_dev, perp_test

def trainAndTest(nGram=3, discount=.5, verbose=True, sampling=False):
    dnames, datas,models = trainModels(nGram, discount, verbose, sampling)
    #######################################    
    # compute the perplexity of all pairs #
    #######################################
   
    
    if not sampling:
        perp_train, perp_dev, perp_test  = perplexityMatrices(dnames, models, datas,nGram)
        #Print perplexities on Train, Dev and Test Sets
        print("-------------------------------")
        print("x train")
        print_table(perp_train, dnames, dnames)
        print("-------------------------------")
        print("x dev")
        print_table(perp_dev, dnames, dnames)
        print("-------------------------------")
        print("x test")
        print_table(perp_test, dnames, dnames)
    
def plotBestHyperParameters(minNGram, maxNGram, discountSizes):

    dnames = ["brown", "reuters", "gutenberg"]
    numModels = len(dnames)
    modelNums = range(numModels)

    nGramHP= list(range(minNGram,maxNGram+1))

    numDiscounts = len(discountSizes)

    perp_dev = defaultdict(lambda:defaultdict(lambda:defaultdict(lambda:[])))


    inOutTypes = ("Within", "Outside")

    def convertIndexToDescription(i):
        d = discountSizes[i % numDiscounts]
        n = nGramHP[i // numDiscounts]
        return f"{n} ngrams with {d} discount"


    datas = getDatas(dnames)
    for m in modelNums:
        print(f"Training Model {dnames[m]}")
        for d in discountSizes:
            model = learn_LM(datas[m], maxNGram, d, False)
            for n in nGramHP:
                print(f"{dnames[m]} {n} grams w/ {d} discount")
                perp_dev["Within"][m][n]+= [model.perplexity(datas[m].test, n)]
                perp_dev["Outside"][m][n]+=  [model.perplexity(datas[m-1].test, n) + model.perplexity(datas[m-2].test, n)]
            model = None

    for type in inOutTypes:
        for m in modelNums:
            pd = sum([perp_dev[type][m][n] for n in nGramHP],[])
            print(f"Min Perplexity {type} "+dnames[m]+" Corpus:", convertIndexToDescription(np.argmin(pd)), min(pd))


    plt.ioff()
    for type in inOutTypes:
        plt.figure()
        plt.xlabel('Discount')
        plt.ylabel('Perplexity')
        plt.title(f'Discount and Ngram Size vs {type} Dev Set Perplexity')
        pd = perp_dev[type]

        colors = {3:['m-','r-','b-'],
        4:['m--','r--','b--'],
        5:['m.-','r.-','b.-'],}

        for m in modelNums:
            for n in nGramHP:
                numDiscounts

                plt.plot(discountSizes, pd[m][n], colors[n][m], label=f"{dnames[m]} {n}-grams")

        plt.legend(loc='upper right')
        plt.show()



def sampling():
    trainAndTest(3,.9, True, True)
    

if __name__ == "__main__":
    trainAndTest(3,.9, True)
    #optmizieAdapter(3,.9)
    #optmizieSampler(maxNGram=3, discount=.9)
    #plotBestHyperParameters(3,5, list(map(lambda int: int/10, range(1,11,2))))