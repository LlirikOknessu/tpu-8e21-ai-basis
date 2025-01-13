# Импортируем необходимые библиотеки
from sklearn.neighbors import LocalOutlierFactor
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Загрузка и базовая очистка (удаление дубликатов, пропусков)
df = pd.read_csv('Kostin/data/raw/insurance.csv')
df = df.drop_duplicates().dropna()

# Удаление выбросов по стандартным отклонениям
def remove_outliers(data, columns, z_thresh=2.5):
    for col in columns:
        mean = data[col].mean()
        std = data[col].std()
        data = data[(data[col] >= mean - z_thresh * std) & (data[col] <= mean + z_thresh * std)]
    return data

numeric_cols = ['age', 'bmi', 'children', 'charges']
df = remove_outliers(df, numeric_cols, z_thresh=2.5)

# Дополнительное обрезание хвостов
for col in numeric_cols:
    q_low = df[col].quantile(0.02)
    q_high = df[col].quantile(0.98)
    df = df[(df[col] >= q_low) & (df[col] <= q_high)]

# Локальный метод для исключения выбросов
X_numeric = df[numeric_cols]
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.05)
outliers = lof.fit_predict(X_numeric)
df = df[outliers == 1]

# Кодирование категорий
df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)

# Разделяем на X и y
X = df_encoded.drop(columns=['charges'])
y = df_encoded['charges']

# Обучение линейной регрессии
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Предсказание и метрики
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse:.2f}")
print(f"R2 Score: {r2:.2f}")

# Визуализация
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Реальные vs Предсказанные значения', fontsize=16)
plt.xlabel('Реальные значения', fontsize=14)
plt.ylabel('Предсказанные значения', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Kostin/data/improved_regression_results_stricter.png', dpi=300)
plt.show()
