import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os.path


dataset_names = ['CIFAR10', 'EMNIST']
class_sizes = [10, 62]
alphas = [0.1, 1.0, 10.0]
sampling_types = ['r_under', 'r_over', 'smote']
client_ids = [i for i in range(100)]

for i, dataset_name in enumerate(dataset_names):
    class_size = class_sizes[i]
    for alpha in alphas:
        for sampling_type in sampling_types:
            table_orig = np.zeros((class_size, len(client_ids)))
            table_resampled = np.zeros((class_size, len(client_ids)))

            max_v = 0
            for client_id in client_ids:
                c_file = f"_y.{dataset_name}.a{alpha}.{sampling_type}.{client_id}"
                orig_name = f"./csv/{c_file}.orig.csv"
                resampled_name = f"./csv/{c_file}.resampled.csv"

                if os.path.exists(orig_name) and os.path.exists(resampled_name):
                    orig = np.loadtxt(orig_name, delimiter=",")
                    resampled = np.loadtxt(resampled_name, delimiter=",")
                else :
                    break

                K, V = np.unique(orig, return_counts=True)
                d_orig = zip(K, V)
                K, V = np.unique(resampled, return_counts=True)
                d_resampled = zip(K, V)
                # import code
                # code.interact(local=locals())

                for k, v in d_orig:
                    table_orig[int(k),client_id] = v
                    max_v = max(max_v, v)
                for k, v in d_resampled:
                    table_resampled[int(k),client_id] = v
                    max_v = max(max_v, v)

            # plot 그림
            plt.figure(figsize=(14,5))
            plt.subplot(1, 2, 1)
            ax = sns.heatmap(table_orig, cmap='Blues', fmt='g', vmin=0, vmax=max_v)
            ax.collections[0].colorbar.set_label("Number of samples")
            plt.title('origin')
            plt.ylabel('Class No.')
            plt.xlabel('Client ID')
            # plt.xticks([0, 9, 20-1, 30-1, 40-1, 50-1, 60-1], ['0', '10', '20', '30', '40', '50', '60'])
            # plt.yticks([0, 19, 40-1, 60-1, 80-1, 100-1], ['0', '20', '40', '60', '80', '100'])

            plt.subplot(1, 2, 2)
            ax = sns.heatmap(table_resampled, cmap='Blues', fmt='g', vmin=0, vmax=max_v)
            ax.collections[0].colorbar.set_label("Number of samples")
            plt.title('resampled')
            plt.ylabel('Class No.')
            plt.xlabel('Client ID')
            # plt.xticks([0, 9, 20-1, 30-1, 40-1, 50-1, 60-1], ['0', '10', '20', '30', '40', '50', '60'])
            # plt.yticks([0, 19, 40-1, 60-1, 80-1, 100-1], ['0', '20', '40', '60', '80', '100'])

            # plt.tight_layout()
            plt.suptitle(f"[{dataset_name}] ({sampling_type}) alpha={alpha}")
            plt.show()
            plt.savefig(f"./png/_y.{dataset_name}.a{alpha}.{sampling_type}.png")
