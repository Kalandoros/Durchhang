#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.optimize import fsolve
import sympy as sp
from sympy import lambdify
from sympy import solve
import symengine as se
from scipy.optimize import root_scalar
from sympy import symbols, solve, nsolve, cosh, exp, Eq, lambdify, nsolve

# Define the expression whose roots we want to find
def scipy_solver():
    w = 11.059
    s = 22
    length = 22.036704308655285

    func = lambda h: length - ((2 * h) / w) * (math.sinh((w * s) / (2 * h)))

    # Plot it

    # Use the numerical solver to find the roots

    horizontal_force_initial_guess = 6000
    horizontal_force_solution = fsolve(func, horizontal_force_initial_guess)

    print("The solution is tau = %f" % horizontal_force_solution)
    print("at which the value of the expression is %f" % func(horizontal_force_solution))


def sympy_solver():
    w = 11.059
    s = 22
    l = 22.036704308655285

    h = sp.symbols('h', real=True, simplify=True, rational=False)
    horizontal_forece_1 = sp.solve(l - ((2 * h) / w) * (sp.sinh((w * s) / (2 * h))), h)

    print(f'Horzizontal Force: {horizontal_forece_1}')

    # l, h, w, s = sp.symbols('l, h, w, s', real=True)
    # sp.solve(l - ((2 * h) / w) * (sp.sinh((w * s) / (2 * h))), h)


def sympy_solver_combine():
    w = 11.059
    s = 22
    l = 22.036704308655285
    h = sp.symbols('h')

    equ = l - ((2 * h) / w) * (sp.sinh((w * s) / (2 * h)))
    expr_symengine = se.sympify(equ)

    lam_f = lambdify(h, expr_symengine)
    sol1 = solve(expr_symengine, h, dict=True)
    print(sol1)
    #sol2 = root_scalar(lam_f, bracket=[0, 10000])
    #print(sol2)


def calculation_function_factory(module):
    w = 11.059
    s = 22
    l = 22.036704308655285
    h = sp.symbols('h')
    eq: Eq = l - ((2 * h) / w) * (sp.sinh((w * s) / (2 * h)))

    s1, s2 = solve(eq.rewrite(sp.sinh).expand(), h)
    return lambdify((h), s1, modules=module)


calculation_mt = calculation_function_factory("math")

if __name__ == "__main__":
    pass
    # scipy_solver()
    # sympy_solver()
    #sympy_solver_combine()
