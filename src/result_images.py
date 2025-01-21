import matplotlib.pyplot as plt
import numpy as np
import textwrap
import matplotlib as mpl

mpl.rcParams['pdf.fonttype'] = 42  # 确保字体嵌入
mpl.rcParams['ps.fonttype'] = 42

# """Compare Overall Improvement"""
#
# colors = ['r', 'g', 'b']  # 每种方法对应的颜色  # 横坐标是从1到7的整数
# attack_methods = ['MI-FGSM', 'LGA-origin', 'LGA']
# x = [range(1, 6), range(1, 8), range(1, 6), range(1, 8)]
# # mlliw-voc2007
# success_rates1 = {
#     'MI-FGSM': [0.969, 0.942, 0.833, 0.069, 0.000],
#     'LGA-origin': [0.997, 0.992, 0.949, 0.257, 0.015],
#     'LGA': [0.998, 0.995, 0.972, 0.391, 0.033]
# }
# avg_perturbation1 = {
#     'MI-FGSM': 183.141,
#     'LGA-origin': 62.074,
#     'LGA': 32.421
# }
# # mlgcn-nuswide
# success_rates2 = {
#     'MI-FGSM': [1.000, 1.000, 1.000, 1.000, 1.000, 0.918, 0.278 ],
#     'LGA-origin': [1.000, 1.000, 1.000, 1.000, 0.999, 0.909, 0.165],
#     'LGA': [1.000, 1.000, 1.000, 1.000, 1.000, 0.983, 0.650]
# }
#
# avg_perturbation2 = {
#     'MI-FGSM': 178.978,
#     'LGA-origin': 63.536,
#     'LGA': 31.311
# }
# # mlliw-voc2012
# success_rates3 = {
#     'MI-FGSM': [1.000, 1.000, 0.996, 0.954, 0.818],
#     'LGA-origin': [1.000, 1.000, 1.000, 0.984, 0.945],
#     'LGA': [1.000, 1.000, 1.000, 0.998, 0.971]
# }
# avg_perturbation3 = {
#     'MI-FGSM': 184.512,
#     'LGA-origin': 66.644,
#     'LGA': 32.368
# }
# # mlliw-nuswide
# success_rates4 = {
#     'MI-FGSM': [1.000, 1.000, 0.952, 0.69, 0.304, 0.017, 0.000],
#     'LGA-origin': [1.000, 1.000, 0.990, 0.875, 0.594, 0.073, 0.000],
#     'LGA': [1.000, 1.000, 0.999, 0.968, 0.721, 0.202, 0.000]
# }
# avg_perturbation4 = {
#     'MI-FGSM': 179.778,
#     'LGA-origin': 62.054,
#     'LGA': 31.726
# }
#
# fig_name = ['Target model is ML-LIW and \n dataset is voc2007',
#             'Target model is ML-GCN and \n dataset is NUS-WIDE',
#             'Target model is ML-LIW and \n dataset is voc2012',
#             'Target model is ML-LIW and \n dataset is NUS-WIDE']

"""比较剪枝操作和CoT的影响"""

colors = ['r', 'g', 'b']  # 每种方法对应的颜色  # 横坐标是从1到7的整数
attack_methods = ['LGA-origin', 'LGA-w/o-CoT', 'LGA']
# mlliw-voc2012
success_rates1 = {
    'LGA-origin': [1, 1, 1, 0.984, 0.945],
    'LGA-w/o-CoT': [1, 1, 0.999, 0.988, 0.955],
    'LGA': [1.000, 1.000, 1.000, 0.998, 0.97]
}
avg_perturbation1 = {
    'LGA-origin': 66.664,
    'LGA-w/o-CoT': 80.091,
    'LGA': 32.368
}
# mlgcn-nuswide
success_rates2 = {
    'LGA-origin': [1.000, 1.000, 1.000, 1.000, 0.999, 0.909, 0.165],
    'LGA-w/o-CoT': [1.000, 1.000, 1.000, 1.000, 1, 0.963, 0.414],
    'LGA': [1.000, 1.000, 1.000, 1.000, 1.000, 0.983, 0.650]
}

avg_perturbation2 = {
    'LGA-origin': 63.536,
    'LGA-w/o-CoT': 75.843,
    'LGA': 31.311
}
# mlgcn-coco
success_rates3 = {
    'LGA-origin': [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 0.975],
    'LGA-w/o-CoT': [1.000, 1.000, 1, 1, 1, 1, 1],
    'LGA': [1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000]
}
avg_perturbation3 = {
    'LGA-origin': 66.247,
    'LGA-w/o-CoT': 78.136,
    'LGA': 31.842
}
# mlliw-nuswide
success_rates4 = {
    'LGA-origin': [1.000, 1.000, 0.990, 0.875, 0.594, 0.073, 0.000],
    'LGA-w/o-CoT': [1.000, 1.000, 0.990, 0.882, 0.563, 0.124, 0.000],
    'LGA': [1.000, 1.000, 0.999, 0.968, 0.721, 0.202, 0.000]
}
avg_perturbation4 = {
    'LGA-origin': 62.054,
    'LGA-w/o-CoT': 74.129,
    'LGA': 31.726
}

fig_name = ['Target model is ML-LIW and \n dataset is voc2012',
            'Target model is ML-GCN and \n dataset is NUS-WIDE',
            'Target model is ML-LIW and \n dataset is COCO',
            'Target model is ML-LIW and \n dataset is NUS-WIDE']
x = [range(1, 6), range(1, 8), range(1, 8), range(1, 8)]

fig, ax = plt.subplots(2, 2, figsize=[8, 8])
succ_rates = [success_rates1, success_rates2, success_rates3, success_rates4]
avg_pert = [avg_perturbation1, avg_perturbation2, avg_perturbation3, avg_perturbation4]
for i in range(len(succ_rates)):
    row = i //2
    col = i % 2
    for j, method in enumerate(succ_rates[i]):
        ax[row, col].plot(x[i], succ_rates[i][method], color=colors[j], label=method)
        # min_y = min(succ_rates[i][method])
        # ax[row, col].text(1, min_y + i, f'Avg Pert of {method}: {avg_pert[j][method]:.2f}', color=colors[j])
        text = f'Avg Pert of {method}: {avg_pert[j][method]:.3f}'
        ax[row, col].text(0.01, 0.1 * j + 0.1, text, transform=ax[row, col].transAxes, verticalalignment='top', color=colors[j], fontsize='small')
    ax[row, col].set_title(fig_name[i])
    ax[row, col].legend()
    ax[row, col].set_ylabel('Success Rate')
    ax[row, col].set_xlabel('n')
plt.subplots_adjust(wspace=0.4, hspace=0.4)
plt.savefig('./results/combined_plots.pdf', format='pdf', transparent=True)
plt.show()

# plt.figure(figsize=(6, 6))
# # 绘制每种方法的折线图
# for i, method in enumerate(attack_methods):
#     plt.plot(x, success_rates[method], color=colors[i], label=method)
#     # 计算每条线的最高点来放置标注
#     plt.text(1, 0.1 * i, f'Avg Pert of {method}: {avg_perturbation[method]:.2f}', color=colors[i])
#
# # 添加图例
# plt.legend(loc='upper right')
#
# # 设置轴标签
# plt.xlabel('Attack Number')
# plt.ylabel('Success Rate')
# plt.xticks(range(1, 6))
#
# # 显示网格
# plt.grid(True)
#
# # 显示图表
# plt.show()