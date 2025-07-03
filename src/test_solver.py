from timeit import timeit

from sympy import symbols, solve, nsolve, cosh, exp, Eq, lambdify, nsolve

def calculation_function_factory(module):
    H, w, x1, x2, y1, y2, B = symbols('H, w, x1, x2, y1, y2, B')
    eq: Eq = (H/w * (cosh(w*(x1+B)/H)-1) - y1) - \
        (H/w * (cosh(w*(x2+B)/H)-1) - y2)

    s1, s2 = solve(eq.rewrite(exp).expand(), B)
    return lambdify((H, w, x1, x2, y1, y2), s1, modules=module)


calculation_mt = calculation_function_factory("math")
calculation_np = calculation_function_factory("numpy")


def symbolic_solve(H, w, x1, x2, y1, y2):
    B = symbols('B')
    B_estimate = -0.5*(x2-x1)
    return float(nsolve(
        (H/w * (cosh(w*(x1+B)/H)-1) - y1) - (H/w * (cosh(w*(x2+B)/H)-1) - y2),
        B,
        B_estimate,
        quick=True))



calc_m = "calculation_mt(1,2,3,4,5,6)"
calc_n = "calculation_np(1,2,3,4,5,6)"
solv = "symbolic_solve(1,2,3,4,5,6)"

ITER = 1000

t_solve = timeit(solv, setup=setup, number=ITER)
print(t_solve) # >>> 2.0169183000107296 seconds for ITER iterations
t_calc_n = timeit(calc_n, setup=setup, number=ITER)
print(t_calc_n) # >>> 0.013456200002110563 seconds for ITER iterations
t_calc_m = timeit(calc_m, setup=setup, number=ITER)
print(t_calc_m) # >>> 0.00293279999459628 seconds for ITER iterations