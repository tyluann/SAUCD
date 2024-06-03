from scipy.optimize import minimize
from scipy.stats import pearsonr
import numpy as np
import functools
np.seterr(all='raise')

def non_linear(b, x):
    return b[0] * (0.5 - 1 / (1 + np.exp(b[1] * (x-b[2])))) + b[3] * x + b[4]

def objective(b, x, y):
    nlx = non_linear(b, x)
    res = np.sum((nlx - y) ** 2)
    # r = []
    # for i in range(x.shape[0]):
    #     r.append(pearsonr(nlx[i], y[i])[0])
    # res = -sum(r)
    return res

def minimize_b(b0, x0, y0, tol):
    x0min = np.min(x0)
    x0max = np.max(x0)
    cons = (
        {'type': 'ineq', 'fun': lambda b: b[1]*(x0min-b[2])+7},
        {'type': 'ineq', 'fun': lambda b: b[1]*(x0max-b[2])+7},
        {'type': 'ineq', 'fun': lambda b: 7-b[1]*(x0min-b[2])},
        {'type': 'ineq', 'fun': lambda b: 7-b[1]*(x0max-b[2])},
        {'type': 'ineq', 'fun': lambda b: b[3]},
    )
    obj_func = functools.partial(objective, x=x0, y=y0)
    try:
        res = minimize(obj_func, b0, method = 'SLSQP', tol=tol, constraints=cons)
    except Exception as e:
        print(e)
        return minimize_b(b0, x0, y0, tol * 10)
    return res.x

def get_b(x0, y0, ntrials=5):
    n_para = 5
    para = np.zeros(n_para)
    min_obj = objective(para, x0, y0)
    for i in range(ntrials):
        tol = 1e-8
        b0 = np.random.randn(n_para)
        b = minimize_b(b0, x0, y0, tol=tol)
        if objective(b, x0, y0) < min_obj:
            min_obj = objective(b, x0, y0)
            para = b
    return para