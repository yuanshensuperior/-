import pandas as pd
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 读取数据
data = pd.read_csv("data1.csv")

# 定义自变量（特征）和因变量
X = data[['镇静药名称', '性别', '年龄', '体重', '身高']]  # 此处需多次添加更改其他患者基本信息等因素以获取完整结果
y = data['petco200']  # 假设需要预测的结果保存在名为'outcome_variable'的列中

# 添加常数列作为截距
X = sm.add_constant(X)

# 建立多元线性回归模型
model = sm.OLS(y, X).fit()

# 打印回归分析结果
print(model.summary())

# 分析F值
if model.f_pvalue < 0.05:
    print("F值显著，可以拒绝总体回归系数为0的原假设，表明存在线性关系。")
else:
    print("F值不显著，无法拒绝总体回归系数为0的原假设，线性关系不明显。")

# 计算R²值
print("R²值:", model.rsquared)

# 计算VIF
vif = pd.DataFrame()
vif["Features"] = X.columns
vif["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
print(vif)

# 分析自变量的显著性
for i in range(1, len(X.columns)):
    if model.pvalues[i] < 0.05:
        print(f"{X.columns[i]} 显著")
    else:
        print(f"{X.columns[i]} 不显著")

# 对比分析自变量对因变量的影响程度（回归系数B值）
print("回归系数B值：")
print(model.params)

# 可视化预测值和实际值的差异
predicted = model.predict(X)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=predicted, y=y)
plt.xlabel('Predicted Values')
plt.ylabel('Actual Values')
plt.title('Predicted vs Actual')
plt.show()