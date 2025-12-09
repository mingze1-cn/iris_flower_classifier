"""导入库"""
import matplotlib
import matplotlib.pyplot as plt
import warnings
import joblib
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.cluster import contingency_matrix
from sklearn.pipeline import Pipeline

# 强制 Tkinter 后端
matplotlib.use('TkAgg')
# 解决中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
warnings.filterwarnings('ignore')

# 加载鸢尾花数据集
iris = load_iris()
# 提取特征数据
x = iris.data
# 提取标签数据
y = iris.target
# 保存特征名列表
feature_names = iris.feature_names
# 保存类别名列表
target_names = iris.target_names

# 数据预处理
scaler = StandardScaler()
# 标准化特征数据
x_scaler = scaler.fit_transform(x)

# 划分训练集和测试集
x_train, x_test, y_train, y_test = train_test_split(x_scaler, y, test_size=0.3, random_state=42)

# 创建分类模型
models = {
    '向量机': SVC(),
    '随机森林': RandomForestClassifier(),
    '逻辑回归': LogisticRegression()
}

# 遍历分类模型
for name, model in models.items():
    # 训练所有模型
    model.fit(x_train, y_train)
    # 预测
    y_pred = model.predict(x_test)
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)

    # 打印所有模型的准确率
    print(f'{name} 准确率：{accuracy:.2%}')
    # 打印分类报告
    print(classification_report(y_test, y_pred))
    # 打印列联矩阵
    print('列联矩阵：')
    print(contingency_matrix(y_test, y_pred))
    print('\n')

# 创建参数网格
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5, 10]
}

# 创建Pipeline对象
rf_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier())
])

# 创建网格搜索模型
search_grid = GridSearchCV(rf_pipeline, param_grid, cv=5, scoring='accuracy')
# 训练模型
search_grid.fit(x_train, y_train)

# 打印随机森林最佳参数
print(f'随机森林最佳参数：{search_grid.best_params_}')
# 打印随机森林最佳精度
print(f'随机森林最佳精度：{search_grid.best_score_:.2%}')

# 获取网格搜索找到的最佳模型
best_model = search_grid.best_estimator_
# 使用最佳模型进行预测
y_pred_best = best_model.predict(x_test)

# 打印优化后随机森林的最佳精度
print(f'优化后随机森林的最佳精度：{accuracy_score(y_test, y_pred_best):.2%}')
# joblib 序列化
joblib.dump(best_model, 'data/iris_model.pkl')

# 生成列联矩阵
cm = contingency_matrix(y_test, y_pred_best)
# 设置画板大小
plt.figure(figsize=(8, 6))
# 绘制热力图
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
# 设置标题和横纵标签
plt.title('优化随机森林的列联矩阵')
plt.xlabel('预测')
plt.ylabel('实际存在的')
# 显示图像
plt.show()