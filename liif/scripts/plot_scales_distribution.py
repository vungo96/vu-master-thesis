import matplotlib.pyplot as plt
import os
import pickle

# Path to pickle files
scale_freq_path = 'save_test/_train_edsr-baseline-liif_test/scale_freq-epoch-1.pickle'
max_scale_freq_path = 'save_test/_train_edsr-baseline-liif_test/scale_max_freq-epoch-1.pickle'

bins = 10

# Load dictionary from first file created via pickle
with open(scale_freq_path, 'rb') as f:
    scale_freq = pickle.load(f)

# Sort dictionary keys
sorted_keys = sorted(scale_freq.keys())

# Plot distribution of first file as a histogram
plt.hist(sorted_keys, weights=[scale_freq[k] for k in sorted_keys], bins=bins, range=(0,bins))
plt.xlabel('Scale')
plt.ylabel('Frequency')
plt.title('Distribution of Scales')

# Save plot of first file to same directory as pickle file
dir_path = os.path.dirname(scale_freq_path)
plot_path = os.path.join(dir_path, 'scale_distribution.png')
plt.savefig(plot_path)

# Load dictionary from second file created via pickle
with open(max_scale_freq_path, 'rb') as f:
    max_scale_freq = pickle.load(f)

# Sort dictionary keys
sorted_keys = sorted(max_scale_freq.keys())

# Plot distribution of second file as a histogram
plt.clf() # Clear previous plot
plt.hist(sorted_keys, weights=[max_scale_freq[k] for k in sorted_keys], bins=bins, range=(0,bins))
plt.xlabel('Scale')
plt.ylabel('Frequency')
plt.title('Distribution of Max Scales')

# Save plot of second file to same directory as pickle file
plot_path = os.path.join(dir_path, 'max_scale_distribution.png')
plt.savefig(plot_path)
