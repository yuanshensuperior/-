import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt

# 读取数据
data = pd.read_csv('data1.csv')
contingency_table = pd.crosstab(data['呛咳'], data['镇静药名称'])

# 进行卡方检验
chi2, p, dof, expected = chi2_contingency(contingency_table)

# 输出检验结果
print("卡方值:", chi2)
print("p值:", p)
print("自由度:", dof)
print("期望频数:", expected)

# 可视化结果
labels = contingency_table.index
adverse_drug_data = np.array(contingency_table)

fig, ax = plt.subplots()
im = ax.imshow(adverse_drug_data, cmap="YlGn")

# 设置图表标题及横纵坐标标签
ax.set_title('Adverse Reaction vs Drug Used')
ax.set_xlabel('Drug Used')
ax.set_ylabel('Adverse Reaction')

# 添加标签
ax.set_xticks(np.arange(len(contingency_table.columns)))
ax.set_yticks(np.arange(len(contingency_table.index)))
ax.set_xticklabels(contingency_table.columns)
ax.set_yticklabels(labels)

# 在热力图上显示数值
for i in range(len(labels)):
    for j in range(len(contingency_table.columns)):
        text = ax.text(j, i, adverse_drug_data[i, j], ha="center", va="center", color="black")

plt.show()