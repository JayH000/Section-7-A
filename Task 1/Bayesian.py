import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import json
from scipy.special import gamma

def beta_function(M, N):
    return (gamma(M + 1) * gamma(N - M + 1)) / gamma(N + 2)

def posterior_distribution(p, M, N):
    return (p**M * (1 - p)**(N-M)) / beta_function(M, N)

def fisher_information(p, N):
    return N / (p * (1 - p))

def expectation_variance(M, N):
    expectation = (M + 1) / (N + 2)
    variance = ((M + 1) * (N - M + 1)) / ((N + 2)**2 * (N + 3))
    return expectation, variance

def load_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    M = sum(data)  # Count of heads (True values)
    N = len(data)  # Total tosses
    return M, N

# Load datasets
filenames = ["/mnt/data/dataset_1.json", "/mnt/data/dataset_2.json", "/mnt/data/dataset_3.json"]
data_sets = [load_data(f) for f in filenames]

p_vals = np.linspace(0, 1, 1000)

plt.figure(figsize=(10, 6))
for i, (M, N) in enumerate(data_sets):
    posterior_vals = posterior_distribution(p_vals, M, N)
    plt.plot(p_vals, posterior_vals, label=f"Dataset {i+1}: M = {M}")
    
    E_p, Var_p = expectation_variance(M, N)
    print(f"Dataset {i+1}: M = {M}, N = {N}, Expectation = {E_p:.4f}, Variance = {Var_p:.6f}")
    
plt.xlabel("p")
plt.ylabel("Posterior Probability")
plt.title("Posterior Distributions for Different Datasets")
plt.legend()
plt.grid()
plt.show()
