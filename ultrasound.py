import numpy as np
from scipy.stats import norm

def envelope (sig , klen ):
    b = klen /2
    s = klen /6
    x = np. arange (-b, b, 1)
    ker = norm . pdf (x, 0, s)
    ker = ker / sum ( ker )

    env = np. convolve ( abs ( sig ), ker)
    return env

def moment (sig , order ): #order -th moment of signal
    nom = np. mean(sig ** order)
    den = np. mean(sig ** 2) ** (order / 2)
    return nom / den