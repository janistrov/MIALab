import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns


def format_data(data, label: str, metric: str):
    return data[data['LABEL'] == label][metric].values


def main(csv_files: str, plot_dir: str):
    metric = 'DICE'  # 'DICE' or 'HDRFDST'
    labels = ('WhiteMatter', 'GreyMatter', 'Amygdala', 'Hippocampus', 'Thalamus')

    # load the CSVs. We usually want to compare different methods (e.g. a set of different features), therefore,
    # we load two CSV (for simplicity, it is the same here)
    df_method1 = pd.read_csv(csv_files[0], sep=';')
    df_method2 = pd.read_csv(csv_files[1], sep=';')
    df_method3 = pd.read_csv(csv_files[2], sep=';')
    df_method4 = pd.read_csv(csv_files[3], sep=';')
    df_method5 = pd.read_csv(csv_files[4], sep=';')
    dfs = [df_method1, df_method2, df_method3, df_method4, df_method5]
    # since post-processing is not implemented, delete PP-results:
    for i in range(int(len(dfs))):
        dfs[i] = dfs[i][0:int(len(dfs[i])/2)]

    # normalization methods
    methods = ('None', 'Z-Score', 'White Stripe', 'Hist. Match.', 'FCM')

    all_list = []  # dim 0: labels, dim 1: methods
    for label in labels:
        method_list = []
        for i, df in enumerate(dfs):
            method_list.append(df[df['LABEL'] == label][metric].values)
        all_list.append(np.asarray(method_list))

    data_mat = np.zeros([len(all_list[0][0]), len(methods), len(labels)])
    for meth in range(len(methods)):
        for lab in range(len(labels)):
            data_mat[:, meth, lab] = all_list[lab][meth]

    data = []
    for lab in range(len(labels)):
        data.append(pd.DataFrame(data_mat[:, :, lab], columns=methods).assign(Location=labels[lab]))

    cdf = pd.concat(data)
    mdf = pd.melt(cdf, id_vars=['Location'], var_name=['Methods'])

    ax = sns.boxplot(x="Location", y="value", hue="Methods", data=mdf)
    if metric is 'DICE':
        ax.set_ylim([0, 1])
    else:
        ax.set_ylim([0, 30])  # set lim manually
    plt.grid(color='gray', linestyle='dashed')
    ax.set_axisbelow(True)
    if metric is 'DICE':
        plt.ylabel('DICE coefficient')
    else:
        plt.ylabel('Hausdorff distance (mm)')
    plt.xlabel(' ')
    plt.savefig('./mia-result/boxplots_DICE_zf.png')  # set name manually
    plt.close()


if __name__ == '__main__':
    """The program's entry point.

    Parse the arguments and run the program.
    """
    parser = argparse.ArgumentParser(description='Result plotting.')

    parser.add_argument(
        '--csv_files',
        type=list,
        default=['mia-result/no-results.csv',
                 'mia-result/z-results.csv',
                 'mia-result/ws-results.csv',
                 'mia-result/hm-results.csv',
                 'mia-result/fcm-results.csv'],
        help='Path to the result CSV file.'
    )

    parser.add_argument(
        '--plot_dir',
        type=str,
        default='mia-result/plots',
        help='Path to the plot directory.'
    )

    args = parser.parse_args()
    main(args.csv_files, args.plot_dir)