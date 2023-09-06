import matplotlib.pyplot as plt
import os
import pickle
import tikzplotlib  # Import the tikzplotlib module

import matplotlib.ticker as ticker

# Path to pickle files
#path = 'save/_train_edsr-baseline-lte-variable-input-div2k-plot-distribution_plot_scale_distribution/'
path = 'save/_train_swinir-baseline-lte-variable-input-div2k-final_sample-2304-scale-1toMax-inputs-48-div2k-edge-crop-batch-64-resume/'

bins = 60

# Set default font to Times New Roman
plt.rcParams['font.family'] = 'Times New Roman'

# Set default font size
plt.rcParams['font.size'] = 14

# Load dictionary from first file created via pickle
with open(path + 'scale_freq.pickle', 'rb') as f:
    scale_freq = pickle.load(f)

# Sort dictionary keys
sorted_keys = sorted(scale_freq.keys())

# bins dependent on largest scale
bins = max(scale_freq.keys()) + 5
print('highest scale:', bins - 5)

# Plot distribution of first file as a histogram
plt.hist(sorted_keys, weights=[scale_freq[k] for k in sorted_keys], bins=bins, range=(0,bins), edgecolor='black')
plt.xlabel('Scale')
plt.ylabel('Frequency')
ax = plt.gca()
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-2, 2))
ax.yaxis.set_major_formatter(formatter)
#plt.yscale('log')  # Set y-axis scale to logarithmic

# Save Matplotlib figure to TikZ code using tikzplotlib
tikzplotlib.save(os.path.join(path, 'scale_distribution.tikz'))
