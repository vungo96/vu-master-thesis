import matplotlib.pyplot as plt
import os
import pickle

scales  = ["2", "3", "4", "6", "12", "18", "24", "30"]

metric = "psnr"

offset = 0.05 #0.05

for scale in scales:

    base_path = 'test_curves/metric_lists/'
    tag = "div2k-test-"
    
    base_path += scale + '/'
    tag += scale 

    paths = [base_path + 'eval_results1to4-baseline.pickle',
         base_path + 'eval_results1to4-lsdir.pickle',
         base_path + 'eval_results1toMax-lsdir.pickle',
         #base_path + '/eval_results1toMax-lsdir-exp-dist-12.pickle', 
        # base_path + '/eval_results1to4-baseline-swinir.pickle',
        # base_path + '/eval_results1toMax-lsdir-swinir.pickle',
        # base_path + '/eval_results1toMax-lsdir-cumulative.pickle',
         #base_path + '/eval_results1to4-div8k.pickle',
         #base_path + '/eval_results1toMax-div8k.pickle',
         #base_path + '/eval_results1toMax-lsdir-sample-patch.pickle',
         #base_path + '/eval_results1to4-lsdir-inputs-64-sample-patch.pickle',
         #base_path + '/eval_results1toMax-lsdir-batch-128.pickle',
         base_path + 'eval_results1toMax-lsdir-sample-2304.pickle',
         base_path + 'eval_results1toMax-lsdir-sample-patch-4096.pickle',
         base_path + 'eval_results1toMax-lsdir-sample-patch.pickle',
         ]

    labels = ['1to4-baseline',
            '1to4-lsdir',
            '1toMax-lsdir',
            # '1toMax-lsdir-exp-dist-12',
            #'1to4-baseline-swinir',
            #'1toMax-baseline-swinir',
            # '1toMax-lsdir-cumulative',
            #'1to4-div8k',
            #'1toMax-div8k',
            #'1toMax-lsdir-sample-patch',
            #'1to4-lsdir-inputs-64-sample-patch',
            #'1toMax-lsdir-batch-128',
            '1toMax-lsdir-sample-2304',
            '1toMax-lsdir-sample-patch-4096',
            '1toMax-lsdir-sample-2304',
            ]

    metric_dicts = []

    for path in paths:
        # Load dictionary from first file created via pickle
        with open(path, 'rb') as f:
            metric_dicts.append(pickle.load(f))

    metric_lists = []
    for metric_dict in metric_dicts:
        metric_list = metric_dict[metric]
        metric_lists.append(metric_list)

    # Create a figure and axes
    fig, ax = plt.subplots()

    # Calculate the y-axis limits based on the PSNR values
    y_min = min([min(metric_list) for metric_list in metric_lists]) - offset
    y_max = max([max(metric_list) for metric_list in metric_lists]) + offset

    # Set y-axis limits
    ax.set_ylim([y_min, y_max])

    # Set x and y axis labels
    ax.set_xlabel("Training Iterations e5")
    ax.set_ylabel("PSNR")

    # Set title
    ax.set_title("PSNR Curves Comparison - EDSR-LTE (x" + scale + ")")

    baseline_psnr = round(metric_lists[0][-1], 2)

    # Plot each PSNR list as a curve
    for i, metric_list in enumerate(metric_lists):
        ax.plot(range(1, len(metric_list)+1), metric_list, label=labels[i] + " last-psnr:" + str(round(metric_list[-1], 2)) + " (" 
                + str(round(round(metric_list[-1], 2) - baseline_psnr, 2)) + ")")

    # Set the position of the legend
    ax.legend(loc='lower right', bbox_to_anchor=(0.95, 0.05), borderaxespad=0.0)


    # Save plot of first file to same directory as pickle file
    plot_path = os.path.join('test_curves/' + metric + '_curves', 'curves_' + tag + '.png')
    plt.savefig(plot_path)
