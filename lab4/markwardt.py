
import numpy as np
import matplotlib.pyplot as plt

# Константы для алгоритма
class CONSTANTS:
    eps_1 = 1e-4 
    eps_2 = 1e-4 
    M = 100       
    alpha = 0.3   
    beta = 0.7   
    E = np.eye(2) # единичная матрица для регуляризации


def backtracking_line_search(func, x_k, grad, initial_lambda=10.0):
    lambda_ = initial_lambda
    f_x_k = func(x_k[0], x_k[1])

    while True:
        x_k_new = x_k + grad * lambda_
        f_x_k_new = func(x_k_new[0], x_k_new[1])
        grad_norm_sq = np.dot(grad, grad)

        # условие Армихо
        if f_x_k_new <= f_x_k - CONSTANTS.alpha * lambda_ * grad_norm_sq:
            break

        # сжатие шага
        lambda_ *= CONSTANTS.beta

    return lambda_


def determinant(matrix):
    return np.linalg.det(matrix)


def gradient(func, x, y):
    h = 1e-6  
    df_dx = (func(x + h, y) - func(x - h, y)) / (2 * h)
    df_dy = (func(x, y + h) - func(x, y - h)) / (2 * h)
    return np.array([df_dx, df_dy])


def hessian_matrix(func, x, y):
    h = 1e-6  
    hess = np.zeros((2, 2))
    hess[0, 0] = (func(x + h, y) - 2 * func(x, y) + func(x - h, y)) / (h ** 2)
    hess[1, 1] = (func(x, y + h) - 2 * func(x, y) + func(x, y - h)) / (h ** 2)
    hess[0, 1] = hess[1, 0] = (func(x + h, y + h) - func(x + h, y - h) - 
                               func(x - h, y + h) + func(x - h, y - h)) / (4 * h ** 2)
    return hess


def inverse_matrix(matrix):
    det = np.linalg.det(matrix)
    if np.abs(det) < 1e-6: 
        matrix += np.eye(2) * 1e-6 
    return np.linalg.inv(matrix)


def f(x, y):
    return ((y + 1) ** 2 + x ** 2) * (x ** 2 + (y - 1) ** 2)


def calculate():
    k = 0
    x_k = np.array([-3, -3]) 
    mu = 1  # параметр регуляризации
    trajectory = [x_k]   

    while k < CONSTANTS.M:
        print(f"Итерация: {k}")

        grad = gradient(f, x_k[0], x_k[1])
        grad_norm = np.linalg.norm(grad)

        if grad_norm <= CONSTANTS.eps_1:
            print(f"Итоговая точка: {x_k}")
            break

        while True:
            hess = hessian_matrix(f, x_k[0], x_k[1])

            d_k = -np.dot(inverse_matrix(hess + CONSTANTS.E * mu), grad)

            x_k_new = x_k + d_k
            print(f"Новая точка: {x_k_new}")

            if f(x_k_new[0], x_k_new[1]) < f(x_k[0], x_k[1]):
                k += 1
                x_k = x_k_new
                trajectory.append(x_k)  
                mu /= 2 
                break

            mu *= 2

    return trajectory


def plot_trajectory(trajectory):
    trajectory = np.array(trajectory)
    plt.figure(figsize=(8, 6))
    plt.contourf(
        X, Y, Z, levels=50, cmap="coolwarm", alpha=0.7
    )
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='black', label='Траектория')
    plt.title("Траектория движения (метод Марквардта)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.colorbar(label='f(x, y)')
    plt.savefig("mw_plt.png")


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{:0.9f}'.format})
    X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    Z = f(X, Y)
    trajectory = calculate()
    plot_trajectory(trajectory)

