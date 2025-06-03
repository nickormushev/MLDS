import matplotlib.pyplot as plt
import numpy as np


import numpy as np

# Data points
data = [
    136.06407833099365,
    133.15839862823486,
    144.23781538009644,
    135.03177452087402,
    132.32022166252136,
    144.0507562160492,
    135.43211555480957,
    128.05647587776184,
    128.16837310791016,
    125.61959147453308
]

# Calculate mean and standard error
mean = np.mean(data)
stderr = np.std(data, ddof=1) / np.sqrt(len(data))  # Standard error = standard deviation / sqrt(n-1)

print("Mean:", mean)
print("Standard Error:", stderr)
