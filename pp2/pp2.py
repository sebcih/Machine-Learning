############################################################################
#COMP 136 Programming Project II
#Author: Cihan Sebzeci
#23.10.2017
############################################################################
import numpy as np
import csv
import matplotlib.pyplot as plt
from regfuncs import *


#data
names = ["100-10", "100-100", "1000-100", "crime", "wine", "f3", "f5"]
set_types = ["train", "trainR", "test", "testR"]

datasets = {}
for name in names:
    datasets[name] = {}
    for set_type in set_types:
        rdr = csv.reader(open(set_type + "-" + name + ".csv"), delimiter= ",")
        datasets[name][set_type] = np.matrix([[float(x) for x in row] 
                                                        for row in rdr])

#TASK I: Regularization
plt.rc("font", family = "serif")
(f, axes) = plt.subplots(1, 5, figsize=(15,3))
plt.tight_layout()

for i,name in enumerate(names[:-2]):
    print(">>>Performing regression for <%s>" % name)
    lmds = []
    train_MSEs = []
    test_MSEs = []
    for lmd in range(151):
        w = w_MLE(datasets[name]["train"], datasets[name]["trainR"], lmd)
        train_MSE = MSE(datasets[name]["train"], datasets[name]["trainR"], w)
        test_MSE =  MSE(datasets[name]["test"], datasets[name]["testR"], w)
        lmds.append(lmd)
        train_MSEs.append(train_MSE)
        test_MSEs.append(test_MSE)

    axes[i].set_title(name)
    axes[0].set_xlabel(r"$\lambda$ term")
    axes[0].set_ylabel("MSE")
    axes[i].plot(lmds, train_MSEs, color="g", label="train", linestyle="--")
    axes[i].plot(lmds, test_MSEs, color="b", label="test")

    # optimal regularization term
    optimal = min(zip(test_MSEs, lmds))
    print(">>>BEST LAMBDA: %f" % optimal[1])
    print(">>>BEST TEST MSE: %f" % optimal[0])

    axes[i].scatter(optimal[1], optimal[0], color="r", label="optimal")
    axes[i].legend(loc=4, fontsize='small')

#Plot the original values as well 
axes[0].plot(lmds, [3.78] * len(lmds), color="k", label="original")
axes[1].plot(lmds, [3.78] * len(lmds), color="k", label="original")
axes[2].plot(lmds, [4.015] * len(lmds), color="k", label="original")

plt.savefig("task1.png")
 
#TASK II: Learning Curves
plt.rc("font", family = "serif")
dataset = datasets["1000-100"]
lmds = [1, 27, 100]  #representetive lambdas
train_sizes = range(10, 810, 10)
N = 15
training_set = np.concatenate((dataset["train"], dataset["trainR"]), axis = 1)
for lmd,color in zip(lmds, "rgb"):
    MSEs = []
    for train_size in train_sizes:
        trial_MSEs = []
        for trial in range(N):
            np.random.shuffle(training_set) #shuflle the array pick
            sample = training_set[:train_size]      #first n-elements as sample
            w = w_MLE(sample[:,:-1], sample[:,-1],lmd)
            trial_MSEs.append(MSE(dataset["test"], dataset["testR"], w))

        MSEs.append(float(sum(trial_MSEs))/N)
    print("for lambda = {}, MSE values are:".format(lmd))
    print(MSEs)
    plt.plot(train_sizes, MSEs, color = color, 
                                        label= r"$\lambda = {}$".format(lmd))
    plt.xlabel("training set size")
    plt.title(r"Learning Curves for different $\lambda$ values")
    plt.ylabel("MSE")
    plt.legend(loc= "upper right", fontsize = "small")

plt.savefig("task2.png")
# TASK III: Bayeian Model Selection

for i,name in enumerate(names[:-2]):
    print(">>>Performing  iterative regression for <%s>" % name)
    (w, alpha, beta, steps) = w_MAP(datasets[name]["train"], 
                                    datasets[name]["trainR"], 1, 1)
    test_MSE = MSE(datasets[name]["test"], datasets[name]["testR"], w)
    print(">>>TEST_MSE: %f" % test_MSE)
    print(">>>ALPHA: %e" % alpha)
    print(">>>BETA: %e" % beta)
    print(">>>ALPHA/BETA: %f" % float(alpha/beta))

#TASK IV: Bayesian Model Selection for Parameters And Model Order

plt.rc("font", family = "serif")
(f, axes) = plt.subplots(2, 2)
plt.tight_layout()
degrees = [d for d in range(1, 11)]
for i,name in enumerate(names[-2:]):
    print(">>>Performing polynomial regression for <%s>" % name)
    train_set = np.ones(datasets[name]["train"].shape)
    test_set = np.ones(datasets[name]["test"].shape)
    MSEs = []
    MSEs_linear = []
    log_es = []
    alphas = []
    betas = []
    for d in degrees:
        new_d_train = np.power(datasets[name]["train"], d)
        new_d_test = np.power(datasets[name]["test"], d)
        train_set = np.concatenate((train_set, new_d_train), axis = 1)
        test_set = np.concatenate((test_set, new_d_test), axis = 1)
        (w, alpha, beta, steps) = w_MAP(train_set, datasets[name]["trainR"],
                                                                        1 ,1)
        alphas.append(alpha)
        betas.append(beta)
        MSEs.append(MSE(test_set, datasets[name]["testR"], w))
        log_es.append(float(log_evidence(train_set, datasets[name]["trainR"],
                                                   alpha, beta)))
        w = w_MLE(test_set, datasets[name]["testR"], 0) #non-regular -> lmd = 0
        MSEs_linear.append(MSE(test_set, datasets[name]["testR"], w))

    #pick the alpha and beta from maximum log evidence value
    optimal = max(zip(log_es, zip(alphas, betas)))
    print(">>>BEST LOG EVIDENCE: %f" % optimal[0])
    print(">>>BEST ALPHA: %e" % optimal[1][0])
    print(">>>BEST BETA: %e" % optimal[1][1])
    
    #plotting settings
    axes[i][0].set_title(name + " MSE vs degree")
    axes[i][1].set_title(name + " Log_evidence vs degree")
    axes[i][0].set_xlabel("Degree")
    axes[i][0].set_ylabel("MSE")
    axes[i][1].set_ylabel("Log_evidence")
    axes[i][0].plot(degrees, MSEs_linear, color="g", label="non-regularized",
                                                   linestyle="--")
    axes[i][0].plot(degrees, MSEs, color="b", label = "polynomial")
    axes[i][1].plot(degrees, log_es, color ="r", label = "log_evidence" )
    axes[i][0].legend(loc=1, fontsize="small")
    axes[i][1].legend(loc=1, fontsize="small")

    #printing results
    print("Non-regularized MSE values:")
    print(MSEs_linear)
    print("Polynomial MSE values:")
    print(MSEs)
    print("Log-evidence values:")
    print(log_es)

plt.savefig("task4.png")