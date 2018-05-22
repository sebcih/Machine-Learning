import numpy as np

def MSE(x, t, w):
    return float((1.0/x.shape[0]) * sum((x[i]*w - t[i])**2
                                    for i in range(x.shape[0])))
def w_MLE(x, t, lmbd):
    return (np.linalg.inv((lmbd * np.identity(x.T.shape[0]) + np.dot(x.T, x)))
                                                                 * (x.T * t))
def w_MAP(x, t, alpha, beta, con_threshold = 0.00001):
    a_diff = np.inf
    b_diff = np.inf
    eigvals = [] 
    step = 0
    while (abs(a_diff) > con_threshold and abs(b_diff) > con_threshold ):
        S_n = float(alpha) * np.eye(x.shape[1]) + beta * (x.T * x)
        S_n = np.linalg.inv(S_n)

        M_n = beta * S_n * x.T * t

        if eigvals == []: # calc eigenvals once
            eigvals = np.linalg.eigvals(x.T * x).real

        gamma = sum(eigval/(alpha + eigval) for eigval in eigvals)

        new_alpha = gamma / (M_n.T * M_n)[0]
        a_diff = new_alpha - alpha
        alpha = new_alpha

        new_beta  = float(x.shape[0] - gamma)
        new_beta /= np.linalg.norm(t - (x * M_n)) ** 2
        b_diff = new_beta - beta
        eigvals = [(new_beta/beta) * eigval for eigval in eigvals]
        #this lines scale eigvals for the new matrix
        beta = new_beta

        step += 1
    return (M_n, alpha, beta, step)

def log_evidence(x, t, alpha, beta):
    N = x.shape[0]
    M = x.shape[1]
    S_n_inv = float(alpha) * np.eye(M) + beta * (x.T * x)
    S_n = np.linalg.inv(S_n_inv)
    M_n = beta * S_n * x.T * t

    term1 = (M/2) * np.log(alpha)
    term2 = (N/2) * np.log(beta)
    term3 = (beta / 2) * np.linalg.norm(t - (x * M_n)) ** 2
    term3 += (alpha / 2) * M_n.T * M_n
    term4 = (1/2) * np.log(np.linalg.det(S_n_inv))
    term5 = (N/2) * np.log(2 * np.pi)

    return(term1 + term2 - term3 - term4 - term5)


