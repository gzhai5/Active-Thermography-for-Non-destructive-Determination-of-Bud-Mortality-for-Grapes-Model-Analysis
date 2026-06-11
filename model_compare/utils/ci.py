import numpy as np


# Bootstrapping function for 95% CI
def bootstrap_ci(values, n=1000, ci=95):
    boot_means = [np.mean(np.random.choice(values, len(values), replace=True)) for _ in range(n)]
    lower = np.percentile(boot_means, (100 - ci) / 2)
    upper = np.percentile(boot_means, 100 - (100 - ci) / 2)
    return np.mean(values), lower, upper