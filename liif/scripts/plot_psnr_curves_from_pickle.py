import matplotlib.pyplot as plt
import os
import pickle

scale  = "30"

base_path = 'test_curves/psnr_lists/' + scale

paths = [
         base_path + '/eval_results1to4-baseline.pickle', 
         base_path + '/eval_results1to4-lsdir.pickle', 
         base_path + '/eval_results1toMax-lsdir.pickle',
         base_path + '/eval_results1toMax-lsdir-5m-iterations.pickle',
         ]

labels = [
          '1to4-baseline', 
          '1to4-lsdir',
          '1toMax-lsdir',
          '1toMax-lsdir-5m-iterations',
          ]

tag = "div2k-" + scale

psnr_lists = []

offset = 0.05
k = 1

for path in paths:
    # Load dictionary from first file created via pickle
    with open(path, 'rb') as f:
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
ax.set_xlabel("Training Iterations e5")
ax.set_ylabel("PSNR")

# Set title
ax.set_title("PSNR Curves Comparison - EDSR-LTE (x" + scale + ")")

baseline_psnr = round(psnr_lists[0][-1], 2)

# Plot each PSNR list as a curve
for i, psnr_list in enumerate(psnr_lists):
    ax.plot(range(1, len(psnr_list)+1), psnr_list, label=labels[i] + " last-psnr:" + str(round(psnr_list[-1], 2)) + " (" 
            + str(round(round(psnr_list[-1], 2) - baseline_psnr, 2)) + ")")
# Set the position of the legend
ax.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), borderaxespad=0.0)


# Save plot of first file to same directory as pickle file
plot_path = os.path.join('test_curves/psnr_curves_old', 'curves_' + tag + '.png')
plt.savefig(plot_path)