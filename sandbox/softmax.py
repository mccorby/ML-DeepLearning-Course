"""Softmax."""

scores = [3.0, 1.0, 0.2]

import numpy as np
import math

def softmax(x):
    # x = x * 10
    # If we multiply the values by 10, probabilities get closer to
    # either 0.0 or 1.0

    x = list(map(lambda value: value/10, x))
    """Compute softmax values for each sets of scores in x."""
    return np.exp(x) / np.sum(np.exp(x), axis=0)


print(softmax(scores))

# Plot softmax curves
import matplotlib.pyplot as plt
x = np.arange(-2.0, 6.0, 0.1)
scores = np.vstack([x, np.ones_like(x), 0.2 * np.ones_like(x)])

plt.plot(x, softmax(scores).T, linewidth=2)
plt.show()
