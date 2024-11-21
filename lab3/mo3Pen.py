import math
from tabulate import tabulate 

class CONSTANTS:
    eps = 1e-4 
    r_0 = 1 # начальное значение параметра штрафа 
    C = 8 # для увеличения штрафа
    alpha = 0.1
    beta = 0.8

def vector_mul(vec, num):
    return [v * num for v in vec]

def vector_sub(vec1, vec2):
    return [v1 - v2 for v1, v2 in zip(vec1, vec2)]

def gradient(func, x, y, h=1e-4):
    df_dx = (func(x + h, y) - func(x - h, y)) / (2 * h)
    df_dy = (func(x, y + h) - func(x, y - h)) / (2 * h)
    return [df_dx, df_dy]

def backtracking_line_search(func, x_k, grad, initial_lambda=10.0):
    lambda_ = initial_lambda
    f_x_k = func(x_k[0], x_k[1])

    while True:
        x_k_new = vector_sub(x_k, vector_mul(grad, lambda_))
        f_x_k_new = func(x_k_new[0], x_k_new[1])
        grad_norm_sq = sum(g ** 2 for g in grad)

        if f_x_k_new <= f_x_k - CONSTANTS.alpha * lambda_ * grad_norm_sq:
            break
        lambda_ *= CONSTANTS.beta
    return lambda_

def cauchy_method(func, x):
    x_k = x[:]
    itr = 0
    while itr < 100:
        grad = gradient(func, x_k[0], x_k[1])
        lambda_ = backtracking_line_search(func, x_k, grad)
        x_k_next = vector_sub(x_k, vector_mul(grad, lambda_))

        if abs(func(x_k_next[0], x_k_next[1]) - func(x_k[0], x_k[1])) <= CONSTANTS.eps:
            return x_k_next
        x_k = x_k_next
        itr += 1

    return [0, 0]

def f(x, y):
    return x**2 + y**2 - 8 * x - 14 * y + 5

def g(x, y):
    return 2 * x**2 + 3 * y**2 - 6

def g_vector(x, y):
    return abs(g(x, y)) if g(x, y) != 0 else 0

def g_plus_vector_func_cut(x, y):
    return 0.5 * (g(x, y) + abs(g(x, y))) if g(x, y) > 0 else 0

def helper_func(x, y):
    g = g_vector(x, y)
    return f(x, y) + (CONSTANTS.r_0 / 2) * (g**2 + g_plus_vector_func_cut(x, y)**2)

def penalty_func(x, y):
    g = g_vector(x, y)
    return (CONSTANTS.r_0 / 2) * (g**2 + g_plus_vector_func_cut(x, y)**2)

def calculate():
    k = 0
    x_k = [-4, -2.5] # начальная точка вне области допустимыx значений

    results = []

    while True:
        x_dash_dot = cauchy_method(helper_func, x_k)

        results.append([k+1, x_k[0], x_k[1], x_dash_dot[0], x_dash_dot[1], CONSTANTS.r_0])
        
        x_k = x_dash_dot

        if penalty_func(x_k[0], x_k[1]) <= CONSTANTS.eps: 
            print("\nТаблица результатов:")
            print(tabulate(results, headers=["Итерация", "x_k[0]", "x_k[1]", "x_dash_dot[0]", "x_dash_dot[1]", "r_k"], tablefmt="grid"))
            print(f"\nОкончательная точка: {x_k[0]:8.4f} {x_k[1]:8.4f}")
            print(f"Количество итераций: {k+1}")
            return
        else: 
            CONSTANTS.r_0 *= CONSTANTS.C 
            k += 1

if __name__ == "__main__":
    calculate()
