import matplotlib.pyplot as plt
import numpy as np


def plot_negtive_and_overall():
    x_ticks_list = [0, 0.4]
    labels = ['SGC', 'MLP']
    bad_point = [50.24, 37.92]
    test_point = [61.08, 58.02]

    x = np.arange(len(labels))  # the label locations
    width = 0.08  # the width of the bars

    fig, ax = plt.subplots(figsize=(5, 4))
    plt.rcParams['savefig.dpi'] = 1000          # 图片像素
    plt.rcParams['figure.dpi'] = 1000           # 分辨率
    ax.bar((0 - width / 2, 0.4 - width / 2), test_point, width, label='Overall', hatch='\\')
    ax.bar((0 + width / 2, 0.4 + width / 2), bad_point, width, label='Negative')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Acc@161', fontsize=20)
    ax.set_xticks(x_ticks_list)
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_yticks([0, 20, 40, 60])
    ax.tick_params(labelsize=13)  # 刻度字体大小
    fig.legend(fontsize=16, bbox_to_anchor=(0.71, 1.01), bbox_transform=ax.transAxes, ncol=1, columnspacing=0.1,
               labelspacing=0.2, markerscale=1, shadow=True, borderpad=0.2, handletextpad=0.2)
    fig.set_tight_layout(tight='rect')
    # plt.savefig('sgc_mlp.pdf')

    plt.show()


def plot_negtive_postive_and_overall():
    x_ticks_list = [0, 0.6]
    labels = ['SGC', 'MLP']
    pos_point = [61.85, 59.88]
    bad_point = [50.24, 37.92]
    test_point = [61.08, 58.02]

    x = np.arange(len(labels))  # the label locations
    width = 0.08  # the width of the bars

    fig, ax = plt.subplots(figsize=(5, 4))
    plt.rcParams['savefig.dpi'] = 1000          # 图片像素
    plt.rcParams['figure.dpi'] = 1000           # 分辨率
    ax.bar((0 - width, 0.6 - width), pos_point, width, label='Positive', hatch='.')
    ax.bar((0, 0.6), test_point, width, label='Overall', hatch='\\')
    ax.bar((0 + width, 0.6 + width), bad_point, width, label='Negative')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Acc@161', fontsize=20)
    ax.set_xticks(x_ticks_list)
    ax.set_xticklabels(labels, fontsize=20)
    ax.set_yticks([0, 20, 40, 60])
    ax.tick_params(labelsize=13)  # 刻度字体大小
    fig.legend(fontsize=16, bbox_to_anchor=(0.71, 1.01), bbox_transform=ax.transAxes, ncol=1, columnspacing=0.1,
               labelspacing=0.1, markerscale=1, shadow=True, borderpad=0.1, handletextpad=0.1)
    fig.set_tight_layout(tight='rect')
    plt.savefig('../pic/pos_overall_neg.png')

    plt.show()


# plot_negtive_and_overall()
plot_negtive_postive_and_overall()
