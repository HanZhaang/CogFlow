import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 原始数据
# --------------------------
bin_size = 5
bin_ranges = [(0, 4), (5, 9), (10, 14), (15, 19), (20, 24), (25, 29)]
class_ids = [1, 2, 3]

mean_delta_ade = np.array([
    [[ 0.0000000e+00, -2.2257068e-03, -7.8722101e-04],
     [-2.0687978e-03,  0.0000000e+00, -1.0258638e-02],
     [-1.8513419e-02, -2.1039657e-02,  0.0000000e+00]],

    [[ 0.0000000e+00,  1.8422761e-03,  6.3991334e-05],
     [ 7.1257968e-03,  0.0000000e+00,  1.1028614e-02],
     [ 6.7231897e-04, -7.6063103e-03,  0.0000000e+00]],

    [[ 0.0000000e+00, -1.8855203e-02,  1.0064994e-03],
     [-9.2298808e-03,  0.0000000e+00, -2.7481614e-02],
     [-3.3005914e-03, -1.2916738e-02,  0.0000000e+00]],

    [[ 0.0000000e+00,  2.3420561e-02, -6.5991501e-03],
     [-2.8865136e-02,  0.0000000e+00,  1.6202020e-02],
     [ 2.0199897e-02,  2.4950013e-02,  0.0000000e+00]],

    [[ 0.0000000e+00,  2.8308269e-03, -2.4153477e-02],
     [-5.1254020e-03,  0.0000000e+00, -3.9805174e-03],
     [-7.0212255e-03,  2.2951957e-02,  0.0000000e+00]],

    [[ 0.0000000e+00, -3.8793760e-03, -2.5648300e-03],
     [ 1.0102469e-02,  0.0000000e+00,  2.4369091e-02],
     [ 1.2409534e-02,  1.4441780e-02,  0.0000000e+00]]
])

# 如果需要用 counts 做 mask，可以额外引入，但这里先不做可视化权重
# counts = np.array([...])

# --------------------------
# 画 6 个 3x3 热力图
# --------------------------
num_bins = len(bin_ranges)
fig, axes = plt.subplots(2, 3, figsize=(10, 6))

# 为了颜色可比，对所有 bin 统一 color scale
vmax = np.max(np.abs(mean_delta_ade))
vmin = -vmax

for i in range(num_bins):
    ax = axes[i // 3, i % 3]
    im = ax.imshow(mean_delta_ade[i],
                   vmin=vmin, vmax=vmax,
                   cmap='bwr',  # 蓝-白-红，便于看正负
                   origin='upper')

    # 标题：显示空间区间
    br = bin_ranges[i]
    ax.set_title(f'Bin {i} : [{br[0]}, {br[1]}]')

    # 坐标轴标签
    ax.set_xticks(range(len(class_ids)))
    ax.set_yticks(range(len(class_ids)))
    ax.set_xticklabels(class_ids)
    ax.set_yticklabels(class_ids)
    ax.set_xlabel('True Class')
    ax.set_ylabel('Replaced Class')

    # 在格子中标数值（可选）
    for r in range(3):
        for c in range(3):
            val = mean_delta_ade[i, r, c]
            ax.text(c, r, f'{val:.3f}',
                    ha='center', va='center',
                    fontsize=7, color='black')

# 统一加一个 colorbar
cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8)
cbar.set_label('ΔADE')

plt.suptitle('Spatial Command Tolerance (ΔADE for Class Replacement)', fontsize=14)
# plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
