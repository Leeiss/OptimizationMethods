import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def f(x1, x2):
    return 3 * (x1 - 6) ** 2 + (x2 - 2) ** 2


def plot_function():
    x1 = np.linspace(0, 12, 100)  
    x2 = np.linspace(0, 4, 100)   
    x1, x2 = np.meshgrid(x1, x2) # создаем двумерные массивы для х1 и х2
    z = np.vectorize(f)(x1, x2) # векторизированная версия функции f

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d') # добавляем трехмерную ось в фигуру
    ax.plot_surface(x1, x2, z, cmap='viridis', alpha=0.8) # строим 3д поверхность на основе сетки х1 х2 и значений z

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('f(x1, x2)')
    ax.set_title('График функции f(x1, x2) = 3(x1 - 6)^2 + (x2 - 2)^2')

    plt.savefig('function_plot2.png')

def f_point(x):
    return f(x[0], x[1])

def create_simplex(x0, L, n):
    p_n = L / (n * np.sqrt(2)) * ((np.sqrt(n + 1)) + n - 1)
    q_n = L / (n * np.sqrt(2)) * ((np.sqrt(n + 1)) - 1)
    
    simplex = np.zeros((n + 1, n))
    simplex[0] = x0  
    
    for j in range(1, n + 1):
        simplex[j] = x0 + np.array([p_n if i == j - 1 else q_n for i in range(n)])
    
    return simplex

def calculating_of_f_simplex(simplex):
    return np.array([f_point(simplex[j]) for j in range(simplex.shape[0])], dtype=float)

def simplex_method(x0, L, gamma, epsilon1, epsilon2):
    n = 2  # размерность 
    simplex = create_simplex(x0, L, n)
    values = calculating_of_f_simplex(simplex)
    iteration = 0
    
    while True:
        iteration += 1
        print("\033[94m" + f"Итерация {iteration}:" + "\033[0m")
        print("Симплекс:\n", np.round(simplex, 4))
        print("Значения функции:\n", np.round(values, 4))

        if np.all(np.abs(values - np.roll(values, 1)) <= epsilon2) and np.all(np.linalg.norm(np.diff(simplex, axis=0), axis=1) <= epsilon1):
            print("\nУсловия сходимости выполнены.")
            break

        worst_index = np.argmax(values)
        worst_vertex = simplex[worst_index]

        # отражение
        reflection = (2 / n) * (np.sum(simplex, axis=0) - worst_vertex) - worst_vertex
        reflection_value = f_point(reflection)

        print(f"x~_p [{float(reflection[0]):.4f}, {float(reflection[1]):.4f}], f(x~_p) {float(reflection_value):.4f}")

        if reflection_value > np.max(values):
            print("Возврат к исходному симплексу с сжатием")
            best_index = np.argmin(values)
            best_vertex = simplex[best_index]
            
            for s in range(n + 1):
                if s != best_index:
                    simplex[s] = gamma * best_vertex + (1 - gamma) * simplex[s]
            values = calculating_of_f_simplex(simplex)
        else:
            simplex[worst_index] = reflection
            values[worst_index] = reflection_value

    optimal_index = np.argmin(values)
    optimal_value = values[optimal_index]
    optimal_point = simplex[optimal_index]

    return np.array(optimal_point, dtype=float), float(optimal_value)

def grad_f(x, h=1e-3):
    df_dx1 = (f(x[0] + h, x[1]) - f(x[0]-h, x[1])) / (2 * h)
    df_dx2 = (f(x[0], x[1] + h) - f(x[0], x[1] - h)) / (2 * h)
    
    return np.array([df_dx1, df_dx2], dtype=object)

def gradient_descent(x0, beta, epsilon):
    x_k = np.array([x0[0], x0[1]])
    iterations = 0
    previous_f_value = f_point(x_k)  # в нач точке
    
    while True:
        grad = grad_f(x_k)

        x_k_next = x_k - beta * grad
        current_f_value = f_point(x_k_next)

        function_change = abs(current_f_value - previous_f_value)

        print(f"\033[94mИтерация {iterations + 1}:\033[0m x = [{x_k_next[0]:.4f}, {x_k_next[1]:.4f}], f(x) = {float(current_f_value):.4f}, |f(x_next) - f(x_k)| = {float(function_change):.4f}")
        
        if function_change <= epsilon:
            x_k = x_k_next
            previous_f_value = current_f_value
            iterations += 1
            print("\nВыполнен критерий остановки алгоритма")
            break
        
        x_k = x_k_next
        previous_f_value = current_f_value
        iterations += 1

    return np.array(x_k, dtype=float), float(current_f_value)


if __name__ == "__main__":
    x0 = np.array([1.0, 2.0])  # Начальная точка
    gamma = 0.5  # коэф сжатия
    L = 1.0  # размер симплекса
    epsilon1 = 0.1  
    epsilon2 = 0.1  

    print("\n-----------Симплексный метод--------------")
    optimal_point_simplex, optimal_value_simplex = simplex_method(x0, L, gamma, epsilon1, epsilon2)
    print(f"\nОптимальная точка (симплекс-метод): [{optimal_point_simplex[0]:.4f}, {optimal_point_simplex[1]:.4f}]")
    print(f"Оптимальное значение функции (симплекс-метод): {optimal_value_simplex:.4f}")

    beta = 0.1  
    epsilon = 0.1  

    print("\n--------------------Градиентный спуск------------------------")
    optimal_point_grad, optimal_value_grad = gradient_descent(x0, beta, epsilon)
    print(f"\nОптимальная точка (градиентный спуск): [{optimal_point_grad[0]:.4f}, {optimal_point_grad[1]:.4f}]")
    print(f"Оптимальное значение функции (градиентный спуск): {optimal_value_grad:.4f}")

    plot_function()
