# Cihan Sebzeci
# calculates perplexity given models as dict 
# and data as list
from math import *
def perplexity(model, data):
    N = len(data)
    try:
        return (exp((-1.0/N) * sum([log(model[word]) for word in data])))
    except:
        return float('Inf')
