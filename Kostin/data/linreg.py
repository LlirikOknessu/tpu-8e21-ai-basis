import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

sns.set_theme(color_codes=True)
df_filtered = pd.read_csv('Kostin/data/filtered_insurance.csv')
df_encoded = pd.get_dummies(df_filtered, columns=['sex', 'smoker', 'region'], drop_first=True)
X = df_encoded.drop(columns=['charges'])
y = df_encoded['charges']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.2f}")
print(f"R2 Score: {r2:.2f}") #прошлый раз была хуйня, надо исправить

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Реальные vs Предсказанные значения', fontsize=16)
plt.xlabel('Реальные значения', fontsize=14)
plt.ylabel('Предсказанные значения', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Kostin/data/linear_regression_results.png', dpi=300)
plt.show()

'''
перед коммитом чистить файлы
еще раз почистить вбросовые
по новой линрег сделать
'''