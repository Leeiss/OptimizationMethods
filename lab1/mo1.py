import random
import matplotlib.pyplot as plt
import numpy as np


def plot_function(f, a, b):
    x_values = np.linspace(a, b, 400)
    y_values = [f(x) for x in x_values]
    plt.figure(figsize=(10, 6))
    plt.plot(x_values, y_values, label='f(x) = (x - 6)^2')
    plt.title('График функции f(x)')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.savefig('function_plot.png')

def f(x):
    return (x - 6) ** 2

def monte_carlo_search(f, a, b, epsilon, max_iters=1000):
    x_min = random.uniform(a, b)
    f_min = f(x_min)
    iterations = 0

    for i in range(max_iters):
        x = random.uniform(a, b)
        f_x = f(x)
        iterations += 1

        if f_x < f_min:
            f_min = f_x
            x_min = x

        print(f"Монте-Карло Итерация {i+1}: x = {x:.6f}, f(x) = {f_x:.6f}, Текущий минимум: x_min = {x_min:.6f}, f_min = {f_min:.6f}")

        if abs(f_min) < epsilon:
            break

    return float(x_min), float(f_min), iterations

def powell_search(f, x1, Dx, epsilon, delta, max_iters=10):
    iterations = 0
    for i in range(max_iters):
        x2 = x1 + Dx
        iterations += 1

        if f(x1) > f(x2):
            x3 = x1 + 2 * Dx
        else:
            x3 = x1 - Dx
        if x3 < x1:
            x1, x2, x3 = x3, x1, x2

        
        Fmin = min(f(x1), f(x2), f(x3))
        xmin = [x1, x2, x3][[f(x1), f(x2), f(x3)].index(Fmin)]

        a1 = (f(x2) - f(x1)) / (x2 - x1)
        a2 = (1 / (x3 - x2)) * ((f(x3) - f(x1)) / (x3 - x1) - a1)
        x_bar = ((x2 + x1) / 2) - (a1 / (2 * a2))

        print(f"Пауэлл Итерация {i+1}: x1 = {x1:.2f}, x2 = {x2:.2f}, x3 = {x3:.2f}, f1 = {f(x1):.2f}, f2 = {f(x2):.2f}, f3 = {f(x3):.2f}, x̅ = {x_bar:.2f}, Fmin = {Fmin:.2f}, xmin = {xmin:.2f}")

        if x_bar < x1 or x_bar > x3:
            x1 = x_bar  
            continue

        if abs(Fmin - f(x_bar)) <= epsilon and abs(xmin - x_bar) <= delta:
            return float(x_bar), float(f(x_bar)), iterations

        best_x = x_bar if f(x_bar) < f(xmin) else xmin
        x1 = best_x-Dx

    return float(xmin), float(Fmin), iterations

def f_prime(x, h=1e-3):
    return (f(x + h) - f(x - h)) / (2 * h)

def f_double_prime(x, h=1e-3):
    return (f(x - h) - 2 * f(x) + f(x + h)) / (h ** 2)

def newton_method(f_prime, f_double_prime, x0, epsilon, max_iters=1000):
    x = x0
    iterations = 0
    for i in range(max_iters):
        x_new = x - f_prime(x) / f_double_prime(x) 
        iterations += 1
        
        print(f"Ньютон Итерация {i+1}: x = {x_new}, f'(x) = {f_prime(x)}, f''(x) = {f_double_prime(x)}")
        
        if abs(f_prime(x_new)) < epsilon:
            return float(x_new), float(f(x_new)), iterations
        
        x = x_new 

    return float(x), float(f(x)), iterations


def main():
    a, b = -1, 10
    epsilon = 0.1  
    delta = 0.1
    Dx = 0.1  
    x0 = 0  
    
    plot_function(f, a, b)

    print("\nМетод Монте-Карло:")
    x_min_mc, f_min_mc, iters_mc = monte_carlo_search(f, a, b, epsilon)
    print(f"Монте-Карло: Найден минимум в точке x = {x_min_mc:.6f}, f(x) = {f_min_mc:.6f}, количество итераций: {iters_mc}")
    
    print("\nМетод Пауэлла:")
    x_min_powell, f_min_powell, iters_powell = powell_search(f, x0, Dx, epsilon, delta)
    print(f"Пауэлл: Найден минимум в точке x = {x_min_powell}, f(x) = {f_min_powell}, количество итераций: {iters_powell}")
    
    print("\nМетод Ньютона:")
    x_min_newton, f_min_newton, iters_newton = newton_method(f_prime, f_double_prime, x0, epsilon)
    print(f"Ньютон: Найден минимум в точке x = {x_min_newton}, f(x) = {f_min_newton}, количество итераций: {iters_newton}")
    
    print("\nСравнение методов по числу итераций:")
    print(f"Метод Монте-Карло: {iters_mc} итераций")
    print(f"Метод Пауэлла: {iters_powell} итераций")
    print(f"Метод Ньютона: {iters_newton} итераций")

if __name__ == "__main__":
    main()
