import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

def plot_hist(data, label, bins=[-20000 + 50*i for i in range(400)],
              text_height=0.001, fontsize=12, horizontalalignment="right"):
    _ = plt.hist(data, bins=bins, density=True, alpha=0.5, label=label)
    _ = plt.axvline(data.mean(), color='k', linestyle='dashed', linewidth=1)
    _ = plt.text(data.mean(), text_height, '{}: {:.2f}'.format(label, data.mean()), fontsize=fontsize,
                horizontalalignment=horizontalalignment)