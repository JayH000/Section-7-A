import numpy as np
import matplotlib.pyplot as plt
import json
import requests
from scipy.special import gamma

def beta_function(M, N):
    return (gamma(M + 1) * gamma(N - M + 1)) / gamma(N + 2)

def posterior_distribution(p, M, N):
    return (p**M * (1 - p)**(N-M)) / beta_function(M, N)

def expectation_variance(M, N):
    expectation = (M + 1) / (N + 2)
    variance = ((M + 1) * (N - M + 1)) / ((N + 2)**2 * (N + 3))
    return expectation, variance

def load_data_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        M = sum(data)  # Count of heads (True values)
        N = len(data)  # Total tosses
        return M, N
    else:
        print(f"Failed to fetch data from {url}")
        return None, None

# URLs of the datasets
urls = [
    "https://zhwangs.github.io/UCSB-comp-phys/section7A/dataset_1.json",
    "https://zhwangs.github.io/UCSB-comp-phys/section7A/dataset_2.json",
    "https://zhwangs.github.io/UCSB-comp-phys/section7A/dataset_3.json"
]

data_sets = [load_data_from_url(url) for url in urls]

# Filter out any datasets that failed to load
data_sets = [ds for ds in data_sets if ds[0] is not None]

# Define p values for plotting
p_vals = np.linspace(0, 1, 1000)

# Plot posterior distributions
plt.figure(figsize=(10, 6))
for i, (M, N) in enumerate(data_sets):
    posterior_vals = posterior_distribution(p_vals, M, N)
    plt.plot(p_vals, posterior_vals, label=f"Dataset {i+1}: M = {M}, N = {N}")
    
    E_p, Var_p = expectation_variance(M, N)
    print(f"Dataset {i+1}: M = {M}, N = {N}, Expectation = {E_p:.4f}, Variance = {Var_p:.6f}")
    
plt.xlabel("p")
plt.ylabel("Posterior Probability")
plt.title("Posterior Distributions for Different Datasets")
plt.legend()
plt.grid()
plt.show()
