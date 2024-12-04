# Импортируем нужные библиотеки
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(color_codes=True)

# Загружаем данные
df = pd.read_csv("Kostin/data/raw/insurance.csv")

# Проверяем начало и конец датасета
print(df.head(5))
print("---------------------------")
print(df.tail(5))

# Проверяем типы данных
print("Типы данных в датасете:")
print(df.dtypes)

# Проверяем на дубли
duplicates = df.duplicated()
print(f"Количество дублирующихся строк: {duplicates.sum()}")
df = df[~duplicates]  # Удаляем дубли

# Проверяем пропуски
print("Количество пропусков в данных:")
print(df.isnull().sum())

# Анализ распределений
# Распределение возраста
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], kde=True, bins=20, color='blue')
plt.title("Распределение возраста", fontsize=16)
plt.xlabel("Возраст", fontsize=14)
plt.ylabel("Частота", fontsize=14)
plt.tight_layout()
plt.savefig('Kostin/data/age_distribution.png', dpi=300)

# Распределение целевой переменной charges
plt.figure(figsize=(10, 6))
sns.histplot(df['charges'], kde=True, bins=20, color='green')
plt.title("Распределение медицинских расходов", fontsize=16)
plt.xlabel("Расходы", fontsize=14)
plt.ylabel("Частота", fontsize=14)
plt.tight_layout()
plt.savefig('Kostin/data/charges_distribution.png', dpi=300)

# Корреляционный анализ (только для числовых столбцов)
plt.figure(figsize=(12, 8))
numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation = df[numerical_columns].corr()
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Корреляционная матрица", fontsize=16)
plt.tight_layout()
plt.savefig('Kostin/data/correlation_matrix.png', dpi=300)

# Анализ выбросов
plt.figure(figsize=(10, 6))
sns.boxplot(x=df['charges'], color='orange')
plt.title("Анализ выбросов в медицинских расходах", fontsize=16)
plt.tight_layout()
plt.savefig('Kostin/data/charges_outliers.png', dpi=300)

# Фильтруем выбросы по charges (задерживаем только данные в пределах 1.5 IQR)
Q1 = df['charges'].quantile(0.25)
Q3 = df['charges'].quantile(0.75)
IQR = Q3 - Q1
df_filtered = df[(df['charges'] >= Q1 - 1.5 * IQR) & (df['charges'] <= Q3 + 1.5 * IQR)]

print(f"Размер данных до фильтрации: {df.shape[0]}")
print(f"Размер данных после фильтрации: {df_filtered.shape[0]}")

# Сохраняем отфильтрованные данные
df_filtered.to_csv('Kostin/data/filtered_insurance.csv', index=False)

# Анализ категориальных данных
# Визуализируем зависимость расходов от региона
plt.figure(figsize=(10, 6))
sns.boxplot(x='region', y='charges', data=df, palette='pastel')
plt.title("Распределение медицинских расходов по регионам", fontsize=16)
plt.xlabel("Регион", fontsize=14)
plt.ylabel("Медицинские расходы", fontsize=14)
plt.tight_layout()
plt.savefig('Kostin/data/region_vs_charges.png', dpi=300)

# Зависимость расходов от курения
plt.figure(figsize=(10, 6))
sns.boxplot(x='smoker', y='charges', data=df, palette='coolwarm')
plt.title("Зависимость медицинских расходов от курения", fontsize=16)
plt.xlabel("Курение", fontsize=14)
plt.ylabel("Медицинские расходы", fontsize=14)
plt.tight_layout()
plt.savefig('Kostin/data/smoking_vs_charges.png', dpi=300)

# Закодируем категориальные данные для численного анализа
df_encoded = pd.get_dummies(df, drop_first=True)

# Корреляционная матрица с кодированием
plt.figure(figsize=(12, 8))
correlation_encoded = df_encoded.corr()
sns.heatmap(correlation_encoded, annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Корреляционная матрица (с категориальными переменными)", fontsize=16)
plt.tight_layout()
plt.savefig('Kostin/data/correlation_matrix_encoded.png', dpi=300)

print("Анализ завершён. Данные сохранены в 'filtered_insurance.csv'.")
