import matplotlib.pyplot as plt
import os
import pickle

paths = ['save_test/_train_edsr-baseline-lte-variable-input_sample-2304-scale-1to4-batch-16/', 
         'save_test/_train_edsr-baseline-lte-variable-input3_sample-4096-scale-1toMax-batch-16-scale-mlp/']

psnr_lists = []

second = ""
offset = 0.5
k = 50

for path in paths:
    # Load dictionary from first file created via pickle
    with open(path + 'eval_results' + second + '.pickle', 'rb') as f:
        psnr_lists.append(pickle.load(f))

# filter psnr_lists
psnr_lists_tmp = psnr_lists.copy()
psnr_lists = []

for tmp in psnr_lists_tmp:
    psnr_list = []
    for i, psnr in enumerate(tmp):
        if i % k == 0:
            psnr_list.append(psnr)
    psnr_lists.append(psnr_list)

# Create a figure and axes
fig, ax = plt.subplots()

# Calculate the y-axis limits based on the PSNR values
y_min = min([min(psnr_list) for psnr_list in psnr_lists]) - offset
y_max = max([max(psnr_list) for psnr_list in psnr_lists]) + offset

# Set y-axis limits
ax.set_ylim([y_min, y_max])

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
plot_path = os.path.join(path, 'curves' + second + '.png')
plt.savefig(plot_path)
