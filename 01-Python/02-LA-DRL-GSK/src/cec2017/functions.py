"""
CEC2017 Benchmark Functions - Standalone Python Implementation
==============================================================

Pure Python implementation for testing. For official results, use the 
original CEC2017 C++/MATLAB implementation.

Author: LA-DRL-GSK Research Team
"""

import numpy as np
from typing import Callable, Tuple


# Base Functions
def sphere(x): return np.sum(x**2, axis=1)
def bent_cigar(x): return x[:,0]**2 + 1e6*np.sum(x[:,1:]**2, axis=1)
def zakharov(x):
    d = x.shape[1]
    idx = np.arange(1, d+1)
    t = np.sum(0.5*idx*x, axis=1)
    return np.sum(x**2, axis=1) + t**2 + t**4
def rosenbrock(x): return np.sum(100*(x[:,1:]-x[:,:-1]**2)**2 + (x[:,:-1]-1)**2, axis=1)
def rastrigin(x): return 10*x.shape[1] + np.sum(x**2 - 10*np.cos(2*np.pi*x), axis=1)
def schwefel(x): return 418.9829*x.shape[1] - np.sum(x*np.sin(np.sqrt(np.abs(x))), axis=1)
def griewank(x):
    d = x.shape[1]
    idx = np.arange(1, d+1)
    return 1 + np.sum(x**2, axis=1)/4000 - np.prod(np.cos(x/np.sqrt(idx)), axis=1)
def ackley(x):
    d = x.shape[1]
    return -20*np.exp(-0.2*np.sqrt(np.sum(x**2, axis=1)/d)) - np.exp(np.sum(np.cos(2*np.pi*x), axis=1)/d) + 20 + np.e
def levy(x):
    w = 1 + (x-1)/4
    return np.sin(np.pi*w[:,0])**2 + np.sum((w[:,:-1]-1)**2*(1+10*np.sin(np.pi*w[:,:-1]+1)**2), axis=1) + (w[:,-1]-1)**2*(1+np.sin(2*np.pi*w[:,-1])**2)
def discus(x): return 1e6*x[:,0]**2 + np.sum(x[:,1:]**2, axis=1)
def elliptic(x):
    d = x.shape[1]
    return np.sum(10**(6*np.arange(d)/(d-1))*x**2, axis=1)
def happycat(x):
    d = x.shape[1]
    s = np.sum(x**2, axis=1)
    return np.abs(s-d)**0.25 + (0.5*s + np.sum(x, axis=1))/d + 0.5
def hgbat(x):
    d = x.shape[1]
    s, sx = np.sum(x**2, axis=1), np.sum(x, axis=1)
    return np.abs(s**2-sx**2)**0.5 + (0.5*s+sx)/d + 0.5

def create_rotation(dim, seed=0):
    Q, _ = np.linalg.qr(np.random.RandomState(seed).randn(dim, dim))
    return Q

def create_shift(dim, seed=0):
    return np.random.RandomState(seed).uniform(-80, 80, dim)

class CEC2017Func:
    def __init__(self, base, dim, fid):
        self.base = base
        self.dim = dim
        self.f_opt = 100.0 * fid
        seed = fid*1000 + dim
        self.shift = create_shift(dim, seed)
        self.rot = create_rotation(dim, seed)
    def __call__(self, x):
        x = np.atleast_2d(x).astype(np.float64)
        z = (x - self.shift) @ self.rot.T
        return self.base(z) + self.f_opt

FUNCS = {
    1: bent_cigar, 3: zakharov, 4: rosenbrock, 5: rastrigin,
    6: lambda x: np.sum(x**2-10*np.cos(2*np.pi*x)+10, axis=1),  # Simplified F6
    7: lambda x: np.sum(x**2, axis=1) + 10*(x.shape[1]-np.sum(np.cos(2*np.pi*x), axis=1)),
    8: rastrigin, 9: levy, 10: schwefel,
}

def get_cec2017_function(func_id: int, dim: int) -> Tuple[Callable, float]:
    if func_id == 2:
        raise ValueError("F2 excluded")
    if func_id in FUNCS:
        return CEC2017Func(FUNCS[func_id], dim, func_id), 100.0*func_id
    # Hybrid/Composition - use combinations
    base = [sphere, rosenbrock, rastrigin, griewank, ackley][func_id % 5]
    return CEC2017Func(base, dim, func_id), 100.0*func_id

all_functions = lambda dim: [get_cec2017_function(i, dim)[0] for i in range(1,31) if i!=2]
