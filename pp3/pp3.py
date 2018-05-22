############################################################################
#COMP 136 Programming Project II
#Author: Cihan Sebzeci
#13.11.2017
############################################################################
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

def sigmoid(a):
    try:
        return 1/(1 + exp(-a))
    except:
        if a < 0:
            return 0
        if a > 0:
            return 1

def split_set(dataset, ratio):
    tmp_set = np.concatenate((dataset["fts"], dataset["lbl"]), axis = 1)
    np.random.shuffle(tmp_set)
    test_set = tmp_set[:int(tmp_set.shape[0]*ratio)]
    train_set = tmp_set[int(tmp_set.shape[0]*ratio):]
    return test_set, train_set

def genModel(data):
    class0_data = np.matrix([x for x in data.tolist() if x[-1] == 0.0])
    class1_data = np.matrix([x for x in data.tolist() if x[-1] == 1.0])
    class0_data = class0_data[:,:-1]
    class1_data = class1_data[:,:-1]

    mu1 = np.mean(class0_data, axis = 0).T
    mu2 = np.mean(class1_data, axis = 0).T

    N0 = class0_data.shape[0]
    N1 = class1_data.shape[0]
    N = N0 + N1

    p_c0 = N0 / N
    p_c1 = N1 / N

    S = (p_c0 * np.cov(class0_data.T) + p_c1 * np.cov(class1_data.T))

    try:
        S_inv = np.linalg.inv(S)
    except:
        S_inv = np.linalg.inv(S + np.eye(S.shape[0]) * (10 ** -9))

    w0 = float((-0.5 * (mu1.T * S_inv) * mu1) + (0.5 * (mu2.T * S_inv) * mu2) 
         + np.log(p_c0 / p_c1))  
    w = S_inv * (mu1 - mu2)

    return w0, w

def discModel(data):
    alpha = 0.1
    tol = 10 ** -3
    labels  = data[:,-1]
    phi = np.append(data[:,:-1], np.ones((data.shape[0],1)), axis = 1)

    w_prev = np.zeros([phi.shape[1],1])
    w = IRLS(w_prev, alpha, phi, labels)
    count = 0
    change = 1
    while count < 100 and change > tol:
        w_prev = w
        w = IRLS(w_prev, alpha, phi, labels)
        count += 1
        change = (float(np.linalg.norm(w - w_prev)) /  
                 float(np.linalg.norm(w_prev)))
    w0 = float(w[-1])
    w = np.array(w[:-1])

    return w0, w

def errorRate(w0, w, data):
    accu = 0
    labels  = data[:,-1]
    data = data[:,:-1]
    for i in range(labels.shape[0]):
        predit = sigmoid(float(data[i]* w) + w0) > 0.5
        if (predit == labels[i]):
            accu += 1.0
    return 1 - (accu / len(labels))

def getLearningCurve(data, ratio, model, repeat = 30):

    learnCurve = []
    train_sizes = np.arange(0.2, 1.2, 0.2)

    for i in range(repeat):
        test, train = split_set(data, ratio)
        tmp_perf = []
        for train_size in train_sizes:
            np.random.shuffle(train)
            real_train_data = train[:int(train.shape[0]*train_size)]
            if(model == 0):
                w0, w = genModel(real_train_data)
                tmp_perf.append(1 - errorRate(w0, w, test))
            else:
                w0, w = discModel(real_train_data)
                tmp_perf.append(errorRate(w0, w, test))

        learnCurve.append(np.array(tmp_perf))
    return np.array(learnCurve), (test.shape[0] + train.shape[0]) * (1-ratio) * train_sizes

def plotErrorBar(dataset, title):
    print("Processing dataset: " + title + "...")
    print("Generative Model...")
    learnCurve_g, trainingSizes_g = getLearningCurve(dataset, 1/3, 0)
    print(learnCurve_g)
    print("Discriminative Model...")
    learnCurve_d, trainingSizes_d = getLearningCurve(dataset, 1/3, 1)
    print(learnCurve_d)
 
    plt.errorbar(trainingSizes_g,np.mean(learnCurve_g, axis = 0),yerr=np.std(learnCurve_g, axis = 0),ecolor='r', label = "Generative Model")
    plt.errorbar(trainingSizes_d,np.mean(learnCurve_d, axis = 0),yerr=np.std(learnCurve_d, axis = 0),ecolor='g', label = "Discriminative Model")
    plt.xlabel("Training sizes")
    plt.ylabel("Error Rate")
    plt.grid("on")
    plt.title("dataset: " + title)
    plt.legend(loc="best")
    print("Save image " + title+".png" + " for dataset: " + title + "...")
    plt.savefig(title + '.png')

def testBLS(data):
    print("Test BLS on dataset irls: ")
    true_w = []
    f = open("irlsw.csv")
    for row in f:
        true_w.append(float(row.strip()))
    true_w = np.array(true_w)
    print("Split dataset into training data and test data: ")
    test, train= split_set(data, 1/3)
    print("Apply Generative Model: ")
    w0_g, w_g = genModel(train)
    print("Apply Discriminative Model: ")
    w0_d, w_d = discModel(train)
    print("True w0: ", true_w[0])
    print("w0 from generative: ", w0_g)
    print("w0 from discriminative: ", w0_d)
    print("True w: ", true_w[1:])
    print("w from generative: ", w_g)
    print("w from discriminative: ", w_d)

names = ["A", "B", "usps", "irlstest"]

datasets = {}
for name in names:
    datasets[name] = {}
    rdr = csv.reader(open(name + ".csv"), delimiter = ",")
    datasets[name]["fts"] = np.matrix([[float(x) for x in row] for row in rdr])
    rdr = csv.reader(open("labels-" + name + ".csv"), delimiter = ",")
    datasets[name]["lbl"] = np.matrix([[float(x) for x in row] for row in rdr])

# testBLS(datasets["irlstest"])

# plotErrorBar(datasets["A"],"A") 
# plotErrorBar(datasets["B"], "B")
# plotErrorBar(datasets["usps"], "usps")

def NewtonMethod(data):

    runtimes = []
    w_vals = []
    alpha = 0.1
    tol = 10 ** -3

    print("Processing Newton Method...")

    for dummy in range(3):
        tmp_runtimes = []
        labels  = data[:,-1]
        phi = np.append(data[:,:-1], np.ones((data.shape[0],1)), axis = 1)
        w_prev = np.zeros([phi.shape[1],1])
        start_time = time.clock()
        w = IRLS(w_prev, alpha, phi, labels)
        end_time = time.clock()
        if(dummy == 0):
            w_vals.append(np.array(w))
        count = 0
        change = 1
        while count < 100 and change > tol:
            w_prev = w
            w = IRLS(w_prev, alpha, phi, labels)
            end_time = time.clock()
            count += 1
            change = (float(np.linalg.norm(w - w_prev)) /  
                 float(np.linalg.norm(w_prev)))
            tmp_runtimes.append(end_time - start_time)
            if(dummy == 0):
                w_vals.append(np.array(w))
        runtimes.append(tmp_runtimes)
    return list(np.mean(runtimes, axis=0)), w_vals


def GradientDescent(data, eta = 0.001, alpha = 0.1):
    tol = 10 ** -3
    runtimes = []
    w_vals = []

    print("Processing Gradient Descent...")
    for dummy in range(3):

        tmp_runtimes = [] 

        labels  = data[:,-1]
        phi = np.append(data[:,:-1], np.ones((data.shape[0],1)), axis = 1)

        w_prev = np.zeros([phi.shape[1],1])

        start_time = time.clock()
        
        Y = np.matrix([sigmoid(float(w_prev.T * row.T)) for row in phi]).T

        w_next = w_prev - eta * (np.dot(phi.T, (Y - labels)) + alpha * w_prev)

        end_time = time.clock()
        tmp_runtimes.append(end_time - start_time)
        
        if(dummy == 0):
            w_vals.append(np.array(w_next))

        count = 0
        change = 1

        while(count < 6000 and change > tol):

            w_prev = w_next

            Y = np.matrix([sigmoid(float(w_prev.T * row.T)) for row in phi]).T

            w_next = w_prev - eta * (np.dot(phi.T, (Y - labels)) + alpha * w_prev)
            
            count += 1
            change = (float(np.linalg.norm(w_next - w_prev)) /  
                      float(np.linalg.norm(w_prev)))

            if(count % 60 == 0):
                end_time = time.clock()

                tmp_runtimes.append(end_time - start_time)
                if(dummy == 0):
                    w_vals.append(np.array(w_next))

        runtimes.append(tmp_runtimes)

    return list(np.mean(runtimes, axis = 0)), w_vals


def CompareTwoMethods(dataset, title):
    tmp_set = np.concatenate((dataset["fts"], dataset["lbl"]), axis = 1)
    test = tmp_set[:int(tmp_set.shape[0]*1/3)]
    train = tmp_set[int(tmp_set.shape[0]*1/3):]

    runtimesNM, w_vals_NM = NewtonMethod(train)
    runtimesGD, w_vals_GD = GradientDescent(train)

    print("Calculate Newton Method's performance...")
    error_rates_NM = [errorRate(w_vals_NM[i][0], w_vals_NM[i][:-1], test) 
                                            for i in range(len(w_vals_NM))]
    print(error_rates_NM)
    print("Calculate Gradient Descent's performance...")
    error_rates_GD = [errorRate(w_vals_GD[i][0], w_vals_GD[i][:-1], test) for i in range(len(w_vals_GD))]
    print(error_rates_GD)

    tmp_val = error_rates_NM[-1]
    tmp_l1 = len(error_rates_NM)
    tmp_l2 = len(error_rates_GD)
    for i in range(tmp_l1, tmp_l2):
        runtimesNM.append(runtimesGD[i])
        error_rates_NM.append(tmp_val)


    plt.plot(runtimesNM, error_rates_NM[:-1], 'r*-', label = "Newton Method")
    plt.plot(runtimesGD, error_rates_GD, 'g*-', label = "Gradient Descent")
    plt.legend(loc = "best")
    plt.xlabel("Runtime")
    plt.ylabel("Error rate")
    plt.title("dataset: " + title)
    plt.grid("on")
    plt.savefig("task2"+title+".png")


print("Processing dataset A: ...")
CompareTwoMethods(datasets["A"], "A")
print("Processing dataset USPS: ...")
CompareTwoMethods(datasets["usps"], "usps")
