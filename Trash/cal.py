import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Data: [(rx, ry, angleX, angleY)]
data = [
    (575, 318, 30.0, 25.0), (526, 313, 36.0, 25.0), (482, 307, 42.0, 25.0), (434, 298, 48.0, 25.0),
    (387, 290, 54.0, 25.0), (337, 279, 60.0, 25.0), (287, 271, 66.0, 25.0), (241, 267, 72.0, 25.0),
    (197, 263, 78.0, 25.0), (153, 295, 84.0, 31.0), (114, 290, 90.0, 31.0), (582, 282, 30.0, 31.0),
    (530, 272, 36.0, 31.0), (485, 267, 42.0, 31.0), (438, 259, 48.0, 31.0), (389, 252, 54.0, 31.0),
    (339, 246, 60.0, 31.0), (288, 235, 66.0, 31.0), (241, 275, 72.0, 37.0), (196, 267, 78.0, 37.0),
    (151, 258, 84.0, 37.0), (111, 252, 90.0, 37.0), (591, 245, 30.0, 37.0), (539, 237, 36.0, 37.0),
    (494, 231, 42.0, 37.0), (445, 224, 48.0, 37.0), (394, 218, 54.0, 37.0), (343, 232, 60.0, 43.0),
    (290, 224, 66.0, 43.0), (243, 219, 72.0, 43.0), (196, 213, 78.0, 43.0), (151, 205, 84.0, 43.0),
    (109, 200, 90.0, 43.0), (599, 195, 30.0, 43.0), (547, 191, 36.0, 43.0), (500, 191, 42.0, 43.0),
    (451, 194, 48.0, 49.0), (400, 180, 54.0, 49.0), (346, 172, 60.0, 49.0), (294, 169, 66.0, 49.0),
    (244, 164, 72.0, 49.0), (197, 157, 78.0, 49.0), (150, 151, 84.0, 49.0), (108, 150, 90.0, 49.0),
    (605, 147, 30.0, 49.0), (553, 145, 36.0, 55.0), (505, 138, 42.0, 55.0), (455, 129, 48.0, 55.0),
    (403, 124, 54.0, 55.0), (350, 118, 60.0, 55.0), (297, 116, 66.0, 55.0), (247, 109, 72.0, 55.0),
    (200, 104, 78.0, 55.0), (154, 105, 84.0, 55.0)
]

# Extract rx, ry, angleX, and angleY
rx = np.array([d[0] for d in data])
ry = np.array([d[1] for d in data])
angleX = np.array([d[2] for d in data])
angleY = np.array([d[3] for d in data])

def polyfit2d(x, y, z, degree=4):
    """
    Fits a 2D polynomial of the given degree to the data (x, y, z).
    """
    terms = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            terms.append((x ** i) * (y ** j))
    A = np.vstack(terms).T
    coeffs, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    return coeffs

def evaluate2dpoly(coeffs, x, y, degree=4):
    """
    Evaluates a 2D polynomial with the given coefficients.
    """
    z = np.zeros_like(x, dtype=np.float64)  # Initialize z as float64 to avoid casting issues
    idx = 0
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            z += coeffs[idx] * (x ** i) * (y ** j)
            idx += 1
    return z



# Fit 4th-degree 2D polynomials
coeffs_angleX = polyfit2d(rx, ry, angleX, degree=4)
coeffs_angleY = polyfit2d(rx, ry, angleY, degree=4)

# Generate grid for evaluation
rx_grid, ry_grid = np.meshgrid(np.linspace(min(rx), max(rx), 100),
                               np.linspace(min(ry), max(ry), 100))
angleX_pred = evaluate2dpoly(coeffs_angleX, rx_grid, ry_grid, degree=4)
angleY_pred = evaluate2dpoly(coeffs_angleY, rx_grid, ry_grid, degree=4)

# Residuals
angleX_residuals = angleX - evaluate2dpoly(coeffs_angleX, rx, ry, degree=4)
angleY_residuals = angleY - evaluate2dpoly(coeffs_angleY, rx, ry, degree=4)

# 3D Visualization for angleX
fig = plt.figure(figsize=(16, 8))

# AngleX
ax1 = fig.add_subplot(121, projection='3d')
ax1.scatter(rx, ry, angleX, c='red', label='Data Points')
ax1.plot_surface(rx_grid, ry_grid, angleX_pred, cmap='viridis', alpha=0.7)
ax1.set_title('3D Polynomial Fit: angleX vs rx, ry')
ax1.set_xlabel('rx')
ax1.set_ylabel('ry')
ax1.set_zlabel('angleX')
ax1.legend()

# AngleY
ax2 = fig.add_subplot(122, projection='3d')
ax2.scatter(rx, ry, angleY, c='blue', label='Data Points')
ax2.plot_surface(rx_grid, ry_grid, angleY_pred, cmap='plasma', alpha=0.7)
ax2.set_title('3D Polynomial Fit: angleY vs rx, ry')
ax2.set_xlabel('rx')
ax2.set_ylabel('ry')
ax2.set_zlabel('angleY')
ax2.legend()

plt.tight_layout()
plt.show()

# Plot residuals
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(range(len(angleX_residuals)), angleX_residuals, c='red', label='Residuals')
plt.axhline(0, color='black', linestyle='--')
plt.title('Residuals for angleX')
plt.xlabel('Data Point Index')
plt.ylabel('Residuals')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(range(len(angleY_residuals)), angleY_residuals, c='blue', label='Residuals')
plt.axhline(0, color='black', linestyle='--')
plt.title('Residuals for angleY')
plt.xlabel('Data Point Index')
plt.ylabel('Residuals')
plt.legend()

plt.tight_layout()
plt.show()
# Print the parameters of the polynomials
print("4th-Degree 2D Polynomial Fit for angleX:")
print("angleX = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2 + c6*x^3 + c7*x^2*y + c8*x*y^2 + c9*y^3 + ...")
for i, coeff in enumerate(coeffs_angleX):
    print(f"c{i}: {coeff:.5e}")

print("\n4th-Degree 2D Polynomial Fit for angleY:")
print("angleY = c0 + c1*x + c2*y + c3*x^2 + c4*x*y + c5*y^2 + c6*x^3 + c7*x^2*y + c8*x*y^2 + c9*y^3 + ...")
for i, coeff in enumerate(coeffs_angleY):
    print(f"c{i}: {coeff:.5e}")

