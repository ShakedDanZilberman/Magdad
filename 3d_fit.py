import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Full MEASUREMENTS data (not truncated)
MEASUREMENTS = [(575, 416, 30.0, 20.0), (526, 413, 36.0, 20.0), (482, 410, 42.0, 20.0),
                (434, 407, 48.0, 20.0), (387, 403, 54.0, 20.0), (337, 402, 60.0, 20.0),
                (287, 400, 66.0, 20.0), (241, 398, 72.0, 20.0), (197, 397, 78.0, 20.0),
                (153, 396, 84.0, 20.0), (114, 396, 90.0, 20.0), (582, 394, 30.0, 24.0),
                (530, 389, 36.0, 24.0), (485, 385, 42.0, 24.0), (438, 381, 48.0, 24.0),
                (389, 378, 54.0, 24.0), (339, 375, 60.0, 24.0), (288, 372, 66.0, 24.0),
                (241, 371, 72.0, 24.0), (196, 368, 78.0, 24.0), (151, 369, 84.0, 24.0),
                (111, 364, 90.0, 24.0), (591, 360, 30.0, 28.0), (539, 353, 36.0, 28.0),
                (494, 349, 42.0, 28.0), (445, 345, 48.0, 28.0), (394, 342, 54.0, 28.0),
                (343, 339, 60.0, 28.0), (290, 337, 66.0, 28.0), (243, 335, 72.0, 28.0),
                (196, 333, 78.0, 28.0), (151, 333, 84.0, 28.0), (109, 331, 90.0, 28.0),
                (599, 326, 30.0, 32.0), (547, 318, 36.0, 32.0), (500, 313, 42.0, 32.0),
                (451, 310, 48.0, 32.0), (400, 306, 54.0, 32.0), (346, 303, 60.0, 32.0),
                (294, 299, 66.0, 32.0), (244, 298, 72.0, 32.0), (197, 295, 78.0, 32.0),
                (150, 294, 84.0, 32.0), (108, 291, 90.0, 32.0), (605, 289, 30.0, 36.0),
                (553, 282, 36.0, 36.0), (505, 279, 42.0, 36.0), (455, 275, 48.0, 36.0),
                (403, 272, 54.0, 36.0), (350, 269, 60.0, 36.0), (297, 267, 66.0, 36.0),
                (247, 265, 72.0, 36.0), (200, 262, 78.0, 36.0), (154, 261, 84.0, 36.0),
                (109, 259, 90.0, 36.0), (609, 256, 30.0, 40.0), (557, 251, 36.0, 40.0),
                (509, 246, 42.0, 40.0), (457, 243, 48.0, 40.0), (405, 239, 54.0, 40.0),
                (353, 237, 60.0, 40.0), (298, 234, 66.0, 40.0), (251, 232, 72.0, 40.0),
                (201, 231, 78.0, 40.0), (156, 229, 84.0, 40.0), (111, 227, 90.0, 40.0),
                (611, 226, 30.0, 44.0), (559, 220, 36.0, 44.0), (512, 216, 42.0, 44.0),
                (461, 212, 48.0, 44.0), (409, 209, 54.0, 44.0), (358, 206, 60.0, 44.0),
                (301, 204, 66.0, 44.0), (253, 202, 72.0, 44.0), (204, 199, 78.0, 44.0),
                (158, 199, 84.0, 44.0), (114, 195, 90.0, 44.0), (613, 196, 30.0, 48.0),
                (560, 191, 36.0, 48.0), (511, 187, 42.0, 48.0), (462, 183, 48.0, 48.0),
                (412, 179, 54.0, 48.0), (358, 177, 60.0, 48.0), (305, 174, 66.0, 48.0),
                (255, 173, 72.0, 48.0), (207, 171, 78.0, 48.0), (159, 170, 84.0, 48.0),
                (117, 165, 90.0, 48.0), (613, 164, 30.0, 52.0), (561, 158, 36.0, 52.0),
                (510, 154, 42.0, 52.0), (462, 150, 48.0, 52.0), (413, 147, 54.0, 52.0),
                (360, 144, 60.0, 52.0), (309, 141, 66.0, 52.0), (259, 139, 72.0, 52.0),
                (212, 136, 78.0, 52.0), (165, 135, 84.0, 52.0), (122, 132, 90.0, 52.0),
                (611, 137, 30.0, 56.0), (558, 131, 36.0, 56.0), (511, 128, 42.0, 56.0),
                (463, 124, 48.0, 56.0), (414, 121, 54.0, 56.0), (364, 118, 60.0, 56.0),
                (310, 116, 66.0, 56.0), (260, 114, 72.0, 56.0), (216, 111, 78.0, 56.0),
                (170, 110, 84.0, 56.0), (125, 106, 90.0, 56.0), (607, 103, 30.0, 60.0),
                (557, 98, 36.0, 60.0), (510, 94, 42.0, 60.0), (464, 91, 48.0, 60.0),
                (417, 87, 54.0, 60.0), (366, 85, 60.0, 60.0), (314, 83, 66.0, 60.0),
                (268, 81, 72.0, 60.0), (220, 78, 78.0, 60.0), (176, 77, 84.0, 60.0),
                (134, 71, 90.0, 60.0)]


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
