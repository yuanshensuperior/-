import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
# 读取数据集
data = pd.read_csv('data1.csv')

# 构建逻辑回归模型
X = data[['性别', '年龄', '体重']]  # 此处需多次更改填入需要的自变量以便获取最终结果
Y = data['呛咳']  # 此处需多次更改填入需要的因变量以便获取最终结果因变量
# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 拟合逻辑回归模型
log_reg = LogisticRegression()
log_reg.fit(X_train, Y_train)

#  输出模型参数
print(log_reg.coef_)  # 参数估计值
print(np.exp(log_reg.coef_))  # 指数化系数，得到优势比

#  模型评价
# 在测试集上进行预测
Y_pred = log_reg.predict(X_test)

# 计算准确率
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# 计算混淆矩阵
conf_matrix = confusion_matrix(Y_test, Y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# 输出分类报告
print("Classification Report:")
print(classification_report(Y_test, Y_pred))

# 添加截距项
X = sm.add_constant(X)

# 拟合模型
logit_model = sm.Logit(Y, X)
result = logit_model.fit()

#  对分类因变量分布状况进行描述
print(data['呛咳'].value_counts())

#  似然比卡方检验
print(result.summary())

#  分析模型参数的显著性
print(result.pvalues)

#  分析回归系数B与OR值
odds_ratio = np.exp(result.params)
print(odds_ratio)

#  分析模型预测
predicted_values = result.predict(X)
predicted_classes = (predicted_values > 0.5).astype(int)  # 二分类阈值设定为0.5
confusion_matrix = pd.crosstab(predicted_classes, Y, rownames=['Predicted'], colnames=['Actual'])
print(confusion_matrix)