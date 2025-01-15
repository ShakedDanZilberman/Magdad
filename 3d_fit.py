import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Full MEASUREMENTS data (not truncated)
MEASUREMENTS = [(575, 300, 30.0, 25.0), (526, 297, 36.0, 25.0), (482, 289, 42.0, 25.0), (434, 281, 48.0, 25.0), (387, 275, 54.0, 25.0), (337, 267, 60.0, 25.0), (287, 262, 66.0, 25.0), (241, 259, 72.0, 25.0), (197, 263, 78.0, 25.0), (153, 283, 84.0, 31.0), (114, 273, 90.0, 31.0), (582, 268, 30.0, 31.0), (530, 262, 36.0, 31.0), (485, 257, 42.0, 31.0), (438, 251, 48.0, 31.0),
 (389, 244, 54.0, 31.0), (339, 235, 60.0, 31.0), (288, 233, 66.0, 31.0), (241, 259, 72.0, 37.0), (196, 250, 78.0, 37.0), (151, 247, 84.0, 37.0), (111, 239, 90.0, 37.0), (591, 235, 30.0, 37.0), (539, 231, 36.0, 37.0), (494, 221, 42.0, 37.0), (445, 219, 48.0, 37.0), (394, 216, 54.0, 37.0), (343, 221, 60.0, 43.0), (290, 214, 66.0, 43.0), (243, 207, 72.0, 43.0),
 (196, 204, 78.0, 43.0), (151, 199, 84.0, 43.0), (109, 194, 90.0, 43.0), (599, 191, 30.0, 43.0), (547, 189, 36.0, 43.0), (500, 189, 42.0, 43.0), (451, 180, 48.0, 49.0), (400, 171, 54.0, 49.0), (346, 165, 60.0, 49.0), (294, 162, 66.0, 49.0), (244, 155, 72.0, 49.0), (197, 152, 78.0, 49.0), (150, 151, 84.0, 49.0), (108, 147, 90.0, 49.0), (605, 148, 30.0, 49.0),
 (553, 134, 36.0, 55.0), (505, 126, 42.0, 55.0), (455, 118, 48.0, 55.0), (403, 114, 54.0, 55.0), (350, 111, 60.0, 55.0), (297, 107, 66.0, 55.0), (247, 102, 72.0, 55.0), (200, 102, 78.0, 55.0), (154, 102, 84.0, 55.0)]


def fit_3d_polynomial(x, y, z, degree=3):
    """
    Fits a 3rd-degree 2D polynomial f(x, y) = z using least squares regression.
    """
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)

    # Generate polynomial terms up to the specified degree
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((x ** i) * (y ** j))

    # Combine terms into design matrix A
    A = np.vstack(terms).T

    # Solve the least squares problem to find the coefficients
    coeffs, residuals, rank, s = np.linalg.lstsq(A, z, rcond=None)

    return coeffs
def print_polynomial(coeffs, degree=3, var1='x', var2='y'):
    """
    Prints the polynomial equation given the coefficients.
    """
    terms = []
    term_idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            term = f"{coeffs[term_idx]:.4f}*{var1}^{i}*{var2}^{j}"
            terms.append(term)
            term_idx += 1
    polynomial = " + ".join(terms)
    return polynomial


def evaluate_polynomial(x, y, coeffs, degree=3):
    """
    Evaluates the fitted 2D polynomial at points (x, y).
    """
    x = np.array(x)
    y = np.array(y)
    z = np.zeros_like(x, dtype=float)

    term_idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            z += coeffs[term_idx] * (x ** i) * (y ** j)
            term_idx += 1

    return z


def main():
    # Extract values from MEASUREMENTS
    rx_values = np.array([item[0] for item in MEASUREMENTS])
    ry_values = np.array([item[1] for item in MEASUREMENTS])
    angleX_values = np.array([item[2] for item in MEASUREMENTS])
    angleY_values = np.array([item[3] for item in MEASUREMENTS])

    # Fit the 3D polynomial to the data
    coeffsX = fit_3d_polynomial(rx_values, ry_values, angleX_values, degree=3)
    coeffsY = fit_3d_polynomial(rx_values, ry_values, angleY_values, degree=3)

    # Create a meshgrid for evaluating the polynomial surface
    x_eval, y_eval = np.meshgrid(np.linspace(min(rx_values), max(rx_values), 50),
                                 np.linspace(min(ry_values), max(ry_values), 50))

    # Evaluate the polynomial on the meshgrid
    angleX_eval = evaluate_polynomial(x_eval, y_eval, coeffsX, degree=3)
    angleY_eval = evaluate_polynomial(x_eval, y_eval, coeffsY, degree=3)

    # Visualization
    fig = plt.figure(figsize=(14, 10))

    # Plot angleX data
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(rx_values, ry_values, angleX_values, color='red', label='angleX Data', s=100)
    ax1.plot_surface(x_eval, y_eval, angleX_eval, cmap='viridis', alpha=0.7, edgecolor='none')
    ax1.set_title("3rd-Degree Polynomial Fit for angleX")
    ax1.set_xlabel('rx')
    ax1.set_ylabel('ry')
    ax1.set_zlabel('angleX')
    ax1.legend()

    # Plot angleY data
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(rx_values, ry_values, angleY_values, color='blue', label='angleY Data', s=100)
    ax2.plot_surface(x_eval, y_eval, angleY_eval, cmap='viridis', alpha=0.7, edgecolor='none')
    ax2.set_title("3rd-Degree Polynomial Fit for angleY")
    ax2.set_xlabel('rx')
    ax2.set_ylabel('ry')
    ax2.set_zlabel('angleY')
    ax2.legend()

    plt.show()

    print("Polynomial for angleX:")
    print(print_polynomial(coeffsX, degree=3, var1='rx', var2='ry'))
    print("\nPolynomial for angleY:")
    print(print_polynomial(coeffsY, degree=3, var1='rx', var2='ry'))


if __name__ == '__main__':
    main()
