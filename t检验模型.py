import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, ttest_rel
from pingouin import compute_effsize

# 读取数据
data = pd.read_csv("data2.csv")

# 选择需要进行正态性检验的列
column_to_test = 'sbp00'#此处需多次更改填入需要的数据以便获取最终结果
sample_data = data[column_to_test]

# 执行Shapiro-Wilk检验
statistic, p_value = shapiro(sample_data)

# 显示检验结果
print("Shapiro-Wilk检验结果:")
print("Statistic:", statistic)
print("P-value:", p_value)

# 判断是否符合正态分布
if p_value < 0.05:
    print("P值小于0.05，样本不符合正态分布")
    # 可以进一步检查峰度和偏度
else:
    print("P值大于等于0.05，样本符合正态分布")

# 配对样本T检验
before = data['sbp00']
after = data['sbp00.1']

t_statistic, t_p_value = ttest_rel(before, after)

# 显示T检验结果
print("配对样本T检验结果:")
print("T-statistic:", t_statistic)
print("P-value:", t_p_value)

# 计算效应量
eff_size = compute_effsize(before, after, eftype='cohen')

# 显示效应量
print("效应量（Cohen's d）:", eff_size)

# 可视化
sns.set(style="whitegrid")
plt.figure(figsize=(10, 6))

# 绘制正态分布图
plt.subplot(1, 2, 1)
sns.histplot(sample_data, kde=True, stat="density", linewidth=0)
plt.title('正态分布图')

# 绘制配对样本数据的差异图
plt.subplot(1, 2, 2)
sns.histplot(after - before, kde=True, stat="density", linewidth=0)
plt.title('配对样本差异图')

plt.show()