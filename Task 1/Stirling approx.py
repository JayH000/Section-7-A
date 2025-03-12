import numpy as np
import matplotlib.pyplot as plt
import math

# Define range of N
N_values = np.linspace(1, 10, 100)  # Smooth curve
N_points = np.arange(1, 11)  # Scatter plot

# Compute log(n!) directly
log_factorial = [math.log(math.factorial(n)) for n in N_points]

# Compute Stirling's approximation
stirling_approx = [n * math.log(n) - n + 0.5 * math.log(2 * math.pi * n) for n in N_points]
stirling_curve = N_values * np.log(N_values) - N_values + 0.5 * np.log(2 * np.pi * N_values)

# Compute log(Gamma(n+1))
gamma_log = [math.lgamma(n + 1) for n in N_points]
gamma_curve = [math.lgamma(n + 1) for n in N_values]

# Plot the functions
plt.figure(figsize=(10, 5))
plt.scatter(N_points, log_factorial, color='red', label='log(n!) (scatter)')
plt.plot(N_values, stirling_curve, label='Stirling Approx.', linestyle='dashed')
plt.plot(N_values, gamma_curve, label='log(Gamma(N+1))', linestyle='solid')
plt.xlabel('N')
plt.ylabel('Logarithm Values')
plt.legend()
plt.title("Comparison of log(n!), Stirling's Approximation, and log(Gamma(N+1))")
plt.grid()
plt.show()

# Plot the difference between Stirling's Approximation and log(Gamma(N+1))
diff = [s - g for s, g in zip(stirling_approx, gamma_log)]
plt.figure(figsize=(10, 5))
plt.scatter(N_points, diff, color='blue', label='Difference (Stirling - log(Gamma))')
plt.axhline(0, color='black', linestyle='dashed')
plt.xlabel('N')
plt.ylabel('Difference')
plt.legend()
plt.title("Difference between Stirling's Approximation and log(Gamma(N+1))")
plt.grid()
plt.show()
