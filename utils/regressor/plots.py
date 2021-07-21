import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path
import numpy as np


def plot_recognition_results(start=0, stop=0, labels=(), save_dir=''):  
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    ax = ax.ravel()
    s = ['Train loss', 'Accuracy', 'Thresholds']
    files = glob.glob(str(Path(save_dir) / 'results*.txt'))

    for fi, f in enumerate(files):
        try:
            # (3, epoch)
            results = np.loadtxt(
                f, usecols=[1, 2, 3], ndmin=2).T
            index = np.argmax(results[1])
            n = results.shape[1]  # number of rows
            x = range(start, min(stop, n) if stop else n)
            for i in range(3):
                y = results[i, x]
                if i in [1, 2, 3]:
                    y[y == 0] = np.nan  # dont show zero loss values
                    # y /= y[0]  # normalize
                label = labels[fi] if len(labels) else Path(f).stem
                # ax[i].plot(x, y, marker='.', label=label, linewidth=2, markersize=8)
                ax[i].plot(x, y, label=label, linewidth=2)
                ax[i].scatter(index, y[index], color='r',
                              label='best', marker='*', linewidth=3)

                # ax[i].set_title(s[i])

                # if i in [3, 4, 8, 9]:
                ax[i].set_title(s[i] + f'\n{results[i][index]}')
                # print(results[i][index])
                # if i in [5, 6, 7]:  # share train and val loss y axes
                #     ax[i].get_shared_y_axes().join(ax[i], ax[i - 5])
        except:
            print('Warning: Plotting error for %s, skipping file' % f)

    fig.tight_layout()
    ax[1].legend()
    fig.savefig(Path(save_dir) / 'results.png', dpi=200)


if __name__ == "__main__":
    plot_recognition_results(save_dir='/d/projects/xinye_competition/second_match/Retail_v2_20210721_105735')
