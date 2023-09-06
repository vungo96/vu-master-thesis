import matplotlib.pyplot as plt
import os
import pickle

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
plt.hist(sorted_keys, weights=[scale_freq[k] for k in sorted_keys], bins=bins, range=(0,bins))
#plt.xlabel('Scale')
#plt.ylabel('Frequency')
#plt.title('Distribution of Scales')

ax = plt.gca()
formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_powerlimits((-2, 2))  # Adjust the power limits if needed
ax.yaxis.set_major_formatter(formatter)

# Save plot of first file to same directory as pickle file
plot_path = os.path.join(path, 'scale_distribution.png')
plt.savefig(plot_path)

# Load dictionary from second file created via pickle
with open(path + 'scale_max_freq.pickle', 'rb') as f:
    max_scale_freq = pickle.load(f)

# Sort dictionary keys
sorted_keys = sorted(max_scale_freq.keys())

# bins dependent on largest scale
bins = max(max_scale_freq.keys()) + 5
print('highest scale:', bins - 5)
print('lowest scale:', min(max_scale_freq.keys()))

# Plot distribution of second file as a histogram
plt.clf() # Clear previous plot
plt.hist(sorted_keys, weights=[max_scale_freq[k] for k in sorted_keys], bins=bins, range=(0,bins))
# plt.xlabel('Scale')
# plt.ylabel('Frequency')
# plt.title('Distribution of Max Scales')

ax = plt.gca()
formatter = ticker.ScalarFormatter(useMathText=True)
# formatter.set_powerlimits((-2, 2))  # Adjust the power limits if needed
ax.yaxis.set_major_formatter(formatter)

# Save plot of second file to same directory as pickle file
plot_path = os.path.join(path, 'max_scale_distribution.png')
plt.savefig(plot_path)
