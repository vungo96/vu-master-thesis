import matplotlib.pyplot as plt
import os
import pickle

# Path to pickle files
path = 'save_test/_train_edsr-baseline-lte-variable-input-lsdir2_sample-4096-scale-1toMax-batch-32-inputs-48-lsdir-5M-iterations/'

bins = 60

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
plt.xlabel('Scale')
plt.ylabel('Frequency')
plt.title('Distribution of Scales')

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

# Plot distribution of second file as a histogram
plt.clf() # Clear previous plot
plt.hist(sorted_keys, weights=[max_scale_freq[k] for k in sorted_keys], bins=bins, range=(0,bins))
plt.xlabel('Scale')
plt.ylabel('Frequency')
plt.title('Distribution of Max Scales')

# Save plot of second file to same directory as pickle file
plot_path = os.path.join(path, 'max_scale_distribution.png')
plt.savefig(plot_path)
