import matplotlib.pyplot as plt
import os
import pickle

paths = ['save_test/_train_edsr-baseline-lte-variable-input_test2/', 
         'save_test/_train_edsr-baseline-lte-variable-input_test2/']

psnr_lists = []

for path in paths:
    # Load dictionary from first file created via pickle
    with open(path + 'eval_results.pickle', 'rb') as f:
        psnr_lists.append(pickle.load(f))

# Create a figure and axes
fig, ax = plt.subplots()

# Set x and y axis labels
ax.set_xlabel("Epoch")
ax.set_ylabel("PSNR")

# Set title
ax.set_title("PSNR Curves Comparison")

# Plot each PSNR list as a curve
for i, psnr_list in enumerate(psnr_lists):
    ax.plot(range(1, len(psnr_list)+1), psnr_list, label=paths[i])

# Add a legend
ax.legend()

# Save plot of first file to same directory as pickle file
plot_path = os.path.join(path, 'curves.png')
plt.savefig(plot_path)
