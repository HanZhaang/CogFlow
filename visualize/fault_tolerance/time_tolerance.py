import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# 输入数据（根据你给的）
# --------------------------
deltas = np.array([-15, -12, -9, -6, -3, 0, 3, 6, 9, 12])
class_ids = ['L', 'F', 'R']

mean_delta_ade = np.array([
    [-0.0035123,  0.00742519, -0.00754051,  0.00287539, -0.00113345,
     -0.00471807,  0.00200364, -0.01096659, -0.00685062, -0.00522655],
    [ 0.02067402,  0.0151375 ,  0.00159197,  0.009271  ,  0.02725107,
      0.00269596,  0.0029571 ,  0.01025422,  0.01629309,  0.01492468],
    [-0.00057592, -0.00620411,  0.00116385,  0.00635612,  0.00599814,
      0.01353325,  0.00452189,  0.00452336, -0.00153772,  0.00051434]
])

# --------------------------
# 开始绘图
# --------------------------
plt.figure(figsize=(8, 5))

colors = ['#1f77b4', '#ff7f0e', '#2ca02c']   # 蓝/橙/绿（经典matplotlib配色）
linestyles = ['-', '--', '-.']

for idx, cls in enumerate(class_ids):
    plt.plot(
        deltas,
        mean_delta_ade[idx],
        marker='o',
        linestyle=linestyles[idx],
        color=colors[idx],
        label=f'Class {cls}'
    )

# 参考线（ΔADE = 0）
plt.axhline(0, color='gray', linewidth=1, linestyle='--')

# --------------------------
# 图形美化
# --------------------------
plt.title("Temporal Tolerance: ΔADE vs Time Shift (Δ)", fontsize=14)
plt.xlabel("Time Shift Δ (frames)", fontsize=12)
plt.ylabel("ΔADE", fontsize=12)
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(title="Command Type")

plt.tight_layout()
plt.show()
