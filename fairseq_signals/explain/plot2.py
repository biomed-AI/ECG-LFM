import numpy as np
import matplotlib.pyplot as plt

# 生成2500个点的随机数据作为折线图的数据和重要性数据（这里用随机数代替）
data_points = np.random.rand(2500)
importance_data = np.random.rand(2500)

# 将重要性数据分成几个段落
num_segments = 5
segmented_importance = np.digitize(importance_data, bins=np.linspace(0, 1, num_segments))

# 创建一个颜色映射，用于为不同段落设置不同的颜色
colors = plt.cm.coolwarm(np.linspace(0, 1, num_segments))

# 绘制折线图，根据分段的重要性数据调整段落的颜色
plt.figure(figsize=(12, 6))
prev_segment = segmented_importance[0]
for i in range(1, len(data_points)):
    if segmented_importance[i] != prev_segment:
        plt.plot(range(i-1, i+1), data_points[i-1:i+1], color=colors[prev_segment], linewidth=2)
        prev_segment = segmented_importance[i]

# 添加图例
legend_handles = [plt.Line2D([0], [0], color=colors[i], label=f'Segment {i}') for i in range(num_segments)]
plt.legend(handles=legend_handles, loc='upper right')
plt.title('Line Plot with Segmented Importance')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig('/home/linsy/ECG_FM/code/fairseq-signals2/fairseq_signals/explain/data/test.png')  # 保存为PNG格式图片
plt.show()