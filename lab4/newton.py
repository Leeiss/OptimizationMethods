
import numpy as np
import matplotlib.pyplot as plt

class CONSTANTS:
    eps_1 = 1e-4 
    eps_2 = 1e-4 
    M = 100      
    alpha = 0.3   
    beta = 0.7   


def backtracking_line_search(func, x_k, grad, initial_lambda=1.0):
    #исп условие Армихо для выбора оптимального значения шага
    lambda_ = initial_lambda
    f_x_k = func(x_k[0], x_k[1])

    while True:
        x_k_new = x_k + grad * lambda_
        f_x_k_new = func(x_k_new[0], x_k_new[1])

        grad_norm_sq = np.dot(grad, grad)
        if f_x_k_new <= f_x_k - CONSTANTS.alpha * lambda_ * grad_norm_sq:
            break

        lambda_ *= CONSTANTS.beta

    return lambda_


def determinant(matrix):
    return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]


def gradient(func, x, y):
    h = 1e-6 
    return np.array([
        (func(x + h, y) - func(x - h, y)) / (2 * h),
        (func(x, y + h) - func(x, y - h)) / (2 * h)
    ])


def hessian_matrix(func, x, y):
    h = 1e-6  
    hessian = np.zeros((2, 2))
    # центральные разности
    hessian[0, 0] = (func(x + h, y) - 2 * func(x, y) + func(x - h, y)) / (h * h)
    hessian[0, 1] = hessian[1, 0] = (func(x + h, y + h) - func(x + h, y - h) 
                                     - func(x - h, y + h) + func(x - h, y - h)) / (4 * h * h)
    hessian[1, 1] = (func(x, y + h) - 2 * func(x, y) + func(x, y - h)) / (h * h)

    return hessian


def inverse_matrix(matrix):
    det = determinant(matrix)
    if det == 0:
        raise ValueError("Matrix is singular and cannot be inverted.")
    inv = np.zeros((2, 2))
    inv[0, 0] = matrix[1, 1] / det
    inv[1, 0] = -matrix[1, 0] / det
    inv[0, 1] = -matrix[0, 1] / det
    inv[1, 1] = matrix[0, 0] / det

    return inv

def f(x, y):
    return ((y + 1) ** 2 + x ** 2) * (x ** 2 + (y - 1) ** 2)

def plot_trajectory(X, Y, Z, trajectory):
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, Z, levels=50, cmap="coolwarm", alpha=0.7)
    plt.plot(trajectory[:, 0], trajectory[:, 1], marker='o', color='black', label='Траектория')
    plt.title("Траектория движения (метод Ньютона)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.colorbar(label='f(x, y)')
    plt.savefig("nwt_plt.png")


def calculate():
    k = 0
    x_k = np.array([-3, -3])  
    trajectory = [x_k]   

    while k < CONSTANTS.M: 
        print(f"Итерация: {k}")

        grad = gradient(f, x_k[0], x_k[1]) 
        grad_norm = np.linalg.norm(grad) 

        if grad_norm <= CONSTANTS.eps_1:
            print(f"Итоговая точка: {x_k}")
            break

        hess = hessian_matrix(f, x_k[0], x_k[1]) 
        hess_inv = inverse_matrix(hess)

        if determinant(hess) > 0:
            d_k = -np.dot(hess_inv, grad)
        else:
            d_k = -grad

        t_k = backtracking_line_search(f, x_k, d_k)

        x_k_new = x_k + d_k * t_k
        trajectory.append(x_k_new)
        print(f"Новая точка: {x_k_new}")

        x_norm = np.linalg.norm(x_k_new - x_k)
        f_norm = abs(f(x_k_new[0], x_k_new[1]) - f(x_k[0], x_k[1]))

        if x_norm <= CONSTANTS.eps_2 and f_norm <= CONSTANTS.eps_2:
            print(f"Итоговая точка: {x_k_new}")
            break

        x_k = x_k_new
        k += 1

    trajectory = np.array(trajectory)
    plot_trajectory(X, Y, Z, trajectory)


if __name__ == "__main__":
    np.set_printoptions(formatter={'float': '{:0.9f}'.format})
    X, Y = np.meshgrid(np.linspace(-5, 5, 100), np.linspace(-5, 5, 100))
    Z = f(X, Y)
    calculate()

