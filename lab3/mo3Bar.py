import numpy as np
from tabulate import tabulate 

class CONSTANTS:
    eps = 0.002
    r_0 = 5.0 # значение параметра штрафа 
    C = 4 # для увеличения параметра штрафа
    alpha = 0.1
    beta = 0.8

def vector_mul(vec, num): 
    return [v * num for v in vec]

def vector_sub(vec1, vec2):
    return [v1 - v2 for v1, v2 in zip(vec1, vec2)]

def gradient(func, x, y):
    h = 1e-5
    dfdx = (func(x + h, y) - func(x - h, y)) / (2 * h)
    dfdy = (func(x, y + h) - func(x, y - h)) / (2 * h)
    return [dfdx, dfdy]

def backtracking_line_search(func, x_k, grad, initial_lambda=1.0):
    lam = initial_lambda
    f_x_k = func(*x_k)
    while True:
        x_k_new = vector_sub(x_k, vector_mul(grad, lam))
        f_x_k_new = func(*x_k_new)
        grad_norm_sq = sum(g**2 for g in grad)

        if f_x_k_new <= f_x_k - CONSTANTS.alpha * lam * grad_norm_sq:
            break

        lam *= CONSTANTS.beta
    return lam

def cauchy_method(func, x):
    x_k = x[:]
    itr = 0
    while itr < 100:
        grad = gradient(func, x_k[0], x_k[1])
        lam = backtracking_line_search(func, x_k, grad)
        x_k_next = vector_sub(x_k, vector_mul(grad, lam))

        if abs(func(*x_k_next) - func(*x_k)) <= CONSTANTS.eps:
            return x_k_next
        x_k = x_k_next
        itr += 1
    return [0, 0]

def f(x, y):
    return x**2 + y**2 - 8 * x - 14 * y + 5

def g(x, y):
    g = 2 * x**2 + 3 * y**2 - 6
    return g if g != 0 else 0

def helper_func(x, y):
    return f(x, y) + (-CONSTANTS.r_0 * np.log(-g(x,y)))

def penalty_func(x, y):
    return -CONSTANTS.r_0 * np.log(-g(x,y))

def calculate():
    k = 0
    x_k = [0.8, -0.3] 
    
    results = [] 

    while True:
        x_dash_dot = cauchy_method(helper_func, x_k) #функция безусловного минимума

        results.append([k+1, x_k[0], x_k[1], x_dash_dot[0], x_dash_dot[1], CONSTANTS.r_0])

        x_k = x_dash_dot

        if g(*x_k) <= 0:
            if abs(penalty_func(*x_k)) <= CONSTANTS.eps:
                print("\nТаблица результатов:")
                print(tabulate(results, headers=["Итерация", "x_k[0]", "x_k[1]", "x_dash_dot[0]", "x_dash_dot[1]", "r_k"], tablefmt="grid"))
                print(f"\nОкончательная точка: {x_k[0]:8.5f} {x_k[1]:8.5f}")
                print(f"Количество итераций: {k+1}")
                return
            else:
                CONSTANTS.r_0 /= CONSTANTS.C
                k += 1

if __name__ == "__main__":
    calculate()
