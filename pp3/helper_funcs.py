from math import exp
import time
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore") # to ignore divide by 0 warning in exloring
                                  # gradient descent learning

def sigmoid(a):
    try:
        return 1/(1 + exp(-a))
    except:
        if a < 0:
            return 0
        if a > 0:
            return 1

def random_split(data, ratio):
    np.random.shuffle(data)
    head = data[:int(data.shape[0]*ratio)]
    rest = data[int(data.shape[0]*ratio):]
    return head, rest

def IRLS(w_prev, phi, labels, alpha = 0.1):    
    y = np.matrix([sigmoid(float(w_prev.T * row.T)) for row in phi]).T
    R = np.eye(y.shape[0])
    for i,n in enumerate(R):
        R[i,i] = float(y[i])*(1 - float(y[i]))

    w_next = phi.T * R * phi
    w_next += alpha * np.eye(w_next.shape[0])

    try:
        w_next = np.linalg.inv(w_next)
    except:
        w_next = np.linalg.inv(w_next + np.eye(w_next.shape[0]) * (10 ** -9))
    w_next = w_prev - (w_next * (phi.T * (y - labels) + (alpha * w_prev)))

    return w_next

def discModel(data, tol = 10 ** -3):
    labels  = data[:,-1]
    phi = np.append(data[:,:-1], np.ones((data.shape[0],1)), axis = 1)
    w_prev = np.zeros([phi.shape[1],1])

    w = IRLS(w_prev, phi, labels)
    count = 0
    change = 1
    while count < 100 and change > tol:
        w_prev = w
        w = IRLS(w_prev, phi, labels)
        count += 1
        change = (float(np.linalg.norm(w - w_prev)) /  
                  float(np.linalg.norm(w_prev)))
    w0 = float(w[-1])
    w = np.array(w[:-1])

    return w0, w

def genModel(data):
    class0_data = data[np.where(data[:,-1] == 0)[0],:-1]
    class1_data = data[np.where(data[:,-1] == 1)[0],:-1]

    mu0 = np.matrix(np.mean(class0_data, axis = 0)).T
    mu1 = np.matrix(np.mean(class1_data, axis = 0)).T

    N0 = class0_data.shape[0]
    N1 = class1_data.shape[0]
    N = N0 + N1
    p_c0 = N0/N
    p_c1 = N1/N

    S = (p_c0 * np.cov(class0_data.T) + p_c1 * np.cov(class1_data.T))

    try:
        S_inv = np.linalg.inv(S)
    except:
        S_inv = np.linalg.inv(S + np.eye(S.shape[0]) * (10 ** -9))

    w0 = float((-0.5 * (mu1.T * S_inv) * mu1) + (0.5 * (mu0.T * S_inv) * mu0) 
                + np.log(p_c1 / p_c0))  
    w = S_inv * (mu1 - mu0)

    return w0, w

def errorRate(w0, w, test):
    error = sum(((sigmoid(float(w.T * row[0,:-1].T) + w0) > 0.5) != row[0,-1])                        for row in test) / test.shape[0]
    return error

def GradientDescent(w_prev, phi, labels, alpha = 0.1, eta = 0.001):
    y = np.matrix([sigmoid(float(w_prev.T * row.T)) for row in phi]).T
    w_next = w_prev - eta * (np.dot(phi.T, (y - labels)) + alpha * w_prev)
    return w_next

def getLearningCurve(data, model, ratio = 1/3 , repeat = 30):
    learnCurve = []
    sizes = np.arange(0.2, 1.2, 0.2)

    for i in range(repeat):
        test, train = random_split(data, ratio)
        tmp_perf = []
        for size in sizes:
            train_data, trash = random_split(train, size) 
            if model == 0:
                w0, w = genModel(train_data)
                tmp_perf.append(errorRate(w0, w, test))
            else:
                w0, w = discModel(train_data)
                tmp_perf.append(errorRate(w0, w, test))

        learnCurve.append(np.array(tmp_perf))
    return np.array(learnCurve), (data.shape[0]) * (1-ratio) * sizes

def plotErrorBar(dataset, title):
    print("Processing dataset: " + title)
    print("Generative Model")
    learnCurve_g, trainingSizes_g = getLearningCurve(dataset, 0)
    print(learnCurve_g)
    print("Discriminative Model")
    learnCurve_d, trainingSizes_d = getLearningCurve(dataset, 1)
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
    plt.clf()

def NewtonMethod(data, alpha = 0.1, tol = 10 ** -3):
    print("Processing Newton Method")
    runtimes = []
    w_vals = []

    for i in range(3):
        tmp_runtimes = [] 

        labels  = data[:,-1]
        phi = np.append(data[:,:-1], np.ones((data.shape[0],1)), axis = 1)
        w_prev = np.zeros([phi.shape[1],1])

        start_time = time.clock()
        w = IRLS(w_prev, phi, labels)
        end_time = time.clock()
        tmp_runtimes.append(end_time - start_time)

        if i == 0:
            w_vals.append(w)

        count = 0
        change = 1
        while count < 100 and change > tol:
            w_prev = w
            w = IRLS(w_prev, phi, labels)
            count += 1
            change = (float(np.linalg.norm(w - w_prev)) /  
                      float(np.linalg.norm(w_prev)))
            end_time = time.clock()
            tmp_runtimes.append(end_time - start_time)
            if i == 0:
                w_vals.append(w)
        
        runtimes.append(tmp_runtimes)

    return list(np.mean(runtimes, axis = 0)), w_vals

def GradientDescentMethod(data, tol = 10 ** -3):
    print("Processing Gradient Descent...")
    runtimes = []
    w_vals = []

    for i in range(3):
        tmp_runtimes = [] 

        labels  = data[:,-1]
        phi = np.append(data[:,:-1], np.ones((data.shape[0],1)), axis = 1)
        w_prev = np.zeros([phi.shape[1],1])

        start_time = time.clock()
        w = GradientDescent(w_prev, phi, labels)
        end_time = time.clock()
        tmp_runtimes.append(end_time - start_time)
        
        if i == 0 :
            w_vals.append(w)

        count = 0
        change = 1
        while(count < 6000 and change > tol):
            w_prev = w
            w = GradientDescent(w_prev, phi, labels)
            count += 1
            change = (float(np.linalg.norm(w - w_prev)) /  
                      float(np.linalg.norm(w_prev)))

            if count % 60 == 0:
                end_time = time.clock()
                tmp_runtimes.append(end_time - start_time)
                if i == 0:
                    w_vals.append(w)
        runtimes.append(tmp_runtimes)
    return list(np.mean(runtimes, axis = 0)), w_vals

def CompareTwoMethods(dataset, title):
    test = dataset[:int(dataset.shape[0]*1/3)]
    train = dataset[int(dataset.shape[0]*1/3):]

    runtimesNM, w_vals_NM = NewtonMethod(train)
    runtimesGD, w_vals_GD = GradientDescentMethod(train)

    print("Calculate Newton Method's performance...")
    error_rates_NM = [errorRate(w_vals_NM[i][-1], w_vals_NM[i][:-1], test) 
                                            for i in range(len(w_vals_NM))]
    print(error_rates_NM)

    print("Calculate Gradient Descent's performance...")
    error_rates_GD = [errorRate(w_vals_GD[i][-1], w_vals_GD[i][:-1], test) 
                                              for i in range(len(w_vals_GD))]
    print(error_rates_GD)

    plt.plot(runtimesNM, error_rates_NM, 'r*-', label = "Newton Method")
    plt.plot(runtimesGD, error_rates_GD, 'g*-', label = "Gradient Descent")
    plt.legend(loc = "best")
    plt.xlabel("Runtime (seconds)")
    plt.ylabel("Error rate")
    plt.title("dataset: " + title)
    plt.grid("on")
    plt.savefig("task2"+title+".png")
    plt.clf()


def GradientDescentSensitivity(dataset, tol = 10 ** -3):
    print(r"Compare different fixed etas")
    test = dataset[:int(dataset.shape[0]*1/3)]
    train = dataset[int(dataset.shape[0]*1/3):]

    eta_vals = [0.00001, 0.001, 0.004]

    for eta, color in zip(eta_vals, "rgb"):
        print(eta)
        w_vals = []
        runtimes = []
        error_rates = []

        labels  = train[:,-1]
        phi = np.append(train[:,:-1], np.ones((train.shape[0],1)), axis = 1)
        w_prev = np.zeros([phi.shape[1],1])

        start_time = time.clock()
        w = GradientDescent(w_prev, phi, labels, eta = eta)
        w_vals.append(w)
        end_time = time.clock()
        runtimes.append(end_time - start_time)
        count = 0
        change = 1
        while(count < 6000 and change > tol):
            w_prev = w
            w = GradientDescent(w_prev, phi, labels, eta = eta)
            count += 1
            change = (float(np.linalg.norm(w - w_prev)) /  
                      float(np.linalg.norm(w_prev)))
            end_time = time.clock()
            if count % 60 == 0:
                end_time = time.clock()
                runtimes.append(end_time - start_time)
                w_vals.append(w)
            error_rates = [errorRate(w_vals[i][-1], w_vals[i][:-1], test)
                                            for i in range(len(w_vals))]
        plt.plot(runtimes, error_rates, color = color ,label = r"$\eta = $" + str(eta))
    plt.xlabel("Runtimes")
    plt.ylabel("Error Rate")
    plt.grid("on")
    plt.title("Dataset B, learning rates")
    plt.legend(loc="best")
    plt.savefig("task3_fig3.png")
    plt.clf()


def LFunction(w, y, phi, labels, alpha = 0.1):
    error = -1.0 * np.sum(np.dot(labels.T, np.log(y)) + np.dot((1.0 - labels).T, np.log(1.0 - y))) + 0.5 * alpha * np.dot(w.T,w) 
    return error

def LFunctionPrime(w, y, phi, labels, alpha = 0.1):
    return  ((phi.T * (y - labels)) + alpha * w)

def TestStepCond(w, LPrime, eta, phi, labels, alpha = 0.1):
    mul = np.dot(eta, LPrime)
    w1 = w - mul
    w2 = w

    y1 = np.matrix([sigmoid(float(w1.T * row.T)) for row in phi]).T
    y2 = np.matrix([sigmoid(float(w2.T * row.T)) for row in phi]).T  

    leftH = LFunction(w1, y1, phi, labels, alpha)
    rightH = (LFunction(w2, y2, phi, labels, alpha) - 
              (eta/2.0) * np.sum(np.square(LPrime)))

    return leftH > rightH 

def GradientDescentLineSearch(data, alpha = 0.1, tol = 10 ** -3):    
    runtimes = []
    w_vals = []
    eta = 0.5
    beta = 0.5

    labels  = data[:,-1]
    phi = np.append(data[:,:-1], np.ones((data.shape[0],1)), axis = 1)
    w_prev = np.zeros([phi.shape[1],1])

    start_time = time.clock()
    y = np.matrix([sigmoid(float(w_prev.T * row.T)) for row in phi]).T
    LPrime = LFunctionPrime(w_prev, y, phi, labels)

    count = 0
    while TestStepCond(w_prev, LPrime, eta, phi, labels):
        count += 1
        eta = eta * beta
        if(tmp_count > 20):
            break

    w_next = w_prev - eta * LPrime
    eta = 0.5
    end_time = time.clock()
    runtimes.append(end_time - start_time)
    w_vals.append(w_next)

    count = 0
    change = 1
    while count < 6000 and change > tol :    
        w_prev = w_next
        y = np.matrix([sigmoid(float(w_prev.T * row.T)) for row in phi]).T
        LPrime = LFunctionPrime(w_prev, y, phi, labels)

        tmp_count = 0
        while TestStepCond(w_prev, LPrime, eta, phi, labels, alpha):
            tmp_count += 1
            eta = eta * beta
            if tmp_count > 20:
                break

        w_next = w_prev - eta * LPrime    
        eta = 0.5
        count += 1
        if count % 60 == 0:
            end_time = time.clock()
            runtimes.append(end_time - start_time)
            w_vals.append(w_next)

    return list(runtimes), w_vals

def PlotGradientDescent(dataset,alpha = 0.1):
    print("Running Gradient Descent with Backtrack line search on dataset B..")
    test = dataset[:int(dataset.shape[0]*1/3)]
    train = dataset[int(dataset.shape[0]*1/3):]

    runtimes, w_vals = GradientDescentLineSearch(train)
    error_rates = [errorRate(w_vals[i][-1], w_vals[i][:-1], test) 
                                            for i in range(len(w_vals))]
    print(error_rates)
    
    plt.plot(runtimes, error_rates, 'r*-')
    plt.xlabel("Runtimes")
    plt.ylabel("Error Rate")
    plt.grid("on")
    plt.title("Dataset B, Backtrack Line Search")
    plt.savefig("task3_linesearch.png")

