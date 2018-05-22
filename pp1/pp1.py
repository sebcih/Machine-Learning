############################################################################
#COMP 136 Programming Project I
#Author: Cihan Sebzeci
#2.10.2017
############################################################################

from perp import *
from math import *
import matplotlib.pyplot as plt

#data
train = open("training_data.txt", "r").read().split()
test  = open("test_data.txt", "r").read().split()

vocab = {k: 0 for k in (test + train)} #dict of all words with count 0
K = len(vocab)

#Parameters
alpha_prime = 2
alpha = [alpha_prime for k in range(K)]
alpha_0 = sum(alpha)

#Model Training
#get empty lists to attach our perplexity results results
ML_train  = []
MAP_train = []
PD_train  = []
ML_test   = []
MAP_test  = []
PD_test   = []

sizes = [1/128.0, 1/64.0, 1/16.0, 1/4.0, 1/1.0]
# loop over all different sizes
for size in sizes:              
    exp_vocab = vocab.copy()
    # calc word counts within the initial segment of given size
    train_set_size = (ceil(len(train)*size)) 
    for i in range(int(train_set_size)): 
        exp_vocab[train[i]] = exp_vocab.get(train[i], 0.0) + 1

    #build our models 
    #(enumerate is kind of useless here since our vector is uniform but this
    # way the code will work well with non uniform alphas as well)
    ML  = {word : exp_vocab.get(word) / train_set_size 
                  for word in exp_vocab}
    MAP = {word : ((exp_vocab.get(word) + alpha[k] - 1) / 
                   (train_set_size + alpha_0 - K))
                  for k, word in enumerate(exp_vocab)}
    PD  = {word : ((exp_vocab.get(word) + alpha[k]) /
                   (train_set_size + alpha_0)) 
                  for k, word in enumerate(exp_vocab)}
    #get perps
    ML_train.append(perplexity(ML, train[:train_set_size]))
    MAP_train.append(perplexity(MAP, train[:train_set_size]))
    PD_train.append(perplexity(PD, train[:train_set_size]))
    ML_test.append(perplexity(ML, test))
    MAP_test.append(perplexity(MAP, test))
    PD_test.append(perplexity(PD, test))


#Model Selection
alpha_primes = range(1,11)
train_set_size = int(ceil(len(train)/128))

log_evidence_list = []
PD_test = []

# same as before just getting word counts
exp_vocab = vocab.copy()
for i in range(train_set_size): 
    exp_vocab[train[i]] = exp_vocab.get(train[i], 0.0) + 1

#loop over all alphas
for alpha_prime in alpha_primes:
    alpha = [alpha_prime for k in range(K)]
    alpha_0 = sum(alpha)

    PD  = {word : ((exp_vocab.get(word) + alpha[k]) /
                   (train_set_size + alpha_0)) 
                  for k, word in enumerate(exp_vocab)}

    PD_test.append(perplexity(PD, test))
    log_evidence =  (log(factorial(alpha_0 - 1)) + 
                    sum(log(factorial(exp_vocab.get(word) + alpha[k] - 1)) 
                                        for k, word in enumerate(exp_vocab)) -
                    log(factorial(alpha_0 + train_set_size - 1 )) -
                    sum(log(factorial(k - 1)) for k in alpha))
    log_evidence_list.append(log_evidence)


#data
train = open("pg121.txt.clean", "r").read().split()
pg141 = open("pg141.txt.clean", "r").read().split()
pg1400 = open("pg1400.txt.clean", "r").read().split()

vocab = {k: 0 for k in (train + pg141 + pg1400)}  
K = len(vocab)

#Parameters
alpha_prime = 2
alpha = [alpha_prime for k in range(K)]
alpha_0 = sum(alpha)

exp_vocab = vocab.copy()
# calc word counts within the initial segment of given size
train_set_size = (ceil(len(train)*size)) 
for i in range(int(train_set_size)): 
    exp_vocab[train[i]] = exp_vocab.get(train[i], 0.0) + 1

#build Model
PD  = {word : ((exp_vocab.get(word) + alpha[k]) / (train_set_size + alpha_0)) 
                for k, word in enumerate(exp_vocab)}

PD_tests = [perplexity(PD, pg141), perplexity(PD, pg1400)]

#Now lets delete the words with less than count of 50 in train
train = [word for word in train if exp_vocab.get(word) >= 50]

alpha_prime = 2
alpha = [alpha_prime for k in range(K)]
alpha_0 = sum(alpha)

exp_vocab = vocab.copy()
# calc word counts within the initial segment of given size
train_set_size = (ceil(len(train)*size)) 
for i in range(int(train_set_size)): 
    exp_vocab[train[i]] = exp_vocab.get(train[i], 0.0) + 1

#build Model
PD  = {word : ((exp_vocab.get(word) + alpha[k]) / (train_set_size + alpha_0)) 
                for k, word in enumerate(exp_vocab)}
PD_tests_purged  = [perplexity(PD, pg141), perplexity(PD, pg1400)]
