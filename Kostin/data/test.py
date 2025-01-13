import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
from datetime import datetime

# Модели
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor, CatBoostClassifier

# Метрики
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score, f1_score

sns.set_theme(color_codes=True)

# ======================
# ШАГ 1. ФУНКЦИИ
# ======================

def create_plots_directory():
    """
    Создаёт папку для сохранения графиков с использованием текущего времени в названии.
    Возвращает путь к созданной папке.
    """
    current_time = datetime.now().strftime("%d-%m-%H-%M")
    graphs_dir = f"graphs/{current_time}"
    os.makedirs(graphs_dir, exist_ok=True)
    return graphs_dir

def plot_histogram(data, column, title, save_path):
    """
    Строит гистограмму с KDE для одного числового признака (column).
    Сохраняет без plt.show().
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(data[column], kde=True, bins=20, color='blue')
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()  # закрываем фигуру, чтобы не отображать график

def filter_iqr(data, column, multiplier=1.5):
    """
    Фильтрация данных по межквартильному размаху (IQR).
    Удаляет наблюдения, лежащие за границами [Q1 - multiplier*IQR, Q3 + multiplier*IQR].
    """
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    mask = (data[column] >= Q1 - multiplier * IQR) & (data[column] <= Q3 + multiplier * IQR)
    return data[mask]

def filter_z_score(data, column, threshold=3):
    """
    Фильтрация данных по Z-оценке.
    Удаляет наблюдения, которые имеют Z-оценку выше threshold или ниже -threshold.
    """
    mean_val = data[column].mean()
    std_val = data[column].std()
    z_score = (data[column] - mean_val) / std_val
    return data[abs(z_score) <= threshold]

# ======================
# ШАГ 2. ЗАГРУЗКА И ПЕРВИЧНАЯ ОБРАБОТКА
# ======================

df_original = pd.read_csv("raw/insurance.csv")

# Удалим дубликаты
df_original = df_original.drop_duplicates()

# Удалим пропуски (для демонстрации — обычно можно обрабатывать иначе)
df_original = df_original.dropna()

# ======================
# ШАГ 3. СОХРАНЯЕМ ГРАФИКИ «ДО ОЧИСТКИ»
# ======================
graphs_dir = create_plots_directory()
plot_histogram(df_original, 'charges',
               "Распределение `charges` ДО очистки",
               save_path=f"{graphs_dir}/charges_distribution_before.png")

# ======================
# ШАГ 4. ПОДГОТОВКА ФУНКЦИЙ ДЛЯ ОБУЧЕНИЯ
# ======================

def prepare_regression_data(df):
    """
    Готовим данные к регрессионным моделям.
    Целевой признак: charges
    Возвращает X, y
    """
    df_temp = df.copy()
    df_temp = pd.get_dummies(df_temp, columns=['sex','smoker','region'], drop_first=True)
    
    X = df_temp.drop('charges', axis=1)
    y = df_temp['charges']
    return X, y

def prepare_classification_data(df):
    """
    Готовим данные к классификационным моделям.
    Целевой признак: smoker (0/1).
    Возвращает X, y
    """
    df_temp = df.copy()
    # Переводим smoker в 0/1
    df_temp['smoker'] = df_temp['smoker'].map({'no': 0, 'yes': 1})
    
    # Будем предсказывать 'smoker'
    y = df_temp['smoker']
    df_temp = df_temp.drop('smoker', axis=1)
    
    # Кодируем оставшиеся категориальные
    df_temp = pd.get_dummies(df_temp, columns=['sex','region'], drop_first=True)
    
    # Для классификации smoker мы не удаляем 'charges', но имейте в виду,
    # что предсказывать "курит/не курит" по расходам — тоже возможный вариант.
    X = df_temp
    return X, y

def train_and_evaluate(df, version_name):
    """
    Обучает:
    - Линейную регрессию (Regression)
    - Логистическую регрессию (Classification)
    - Дерево решений (Regression)
    - CatBoost (Regression + Classification)

    Печатает метрики и возвращает словарь с результатами.
    """
    results = {}
    print(f"\n=== Фильтрация: {version_name} ===")

    # ---------------------------
    # 1) ЛИНЕЙНАЯ РЕГРЕССИЯ
    # ---------------------------
    X_reg, y_reg = prepare_regression_data(df)
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    linreg = LinearRegression()
    linreg.fit(X_train_reg, y_train_reg)
    y_pred_lin = linreg.predict(X_test_reg)
    r2_lin = r2_score(y_test_reg, y_pred_lin)
    rmse_lin = mean_squared_error(y_test_reg, y_pred_lin, squared=False)
    print(f"  LinearRegression -> R2: {r2_lin:.3f}, RMSE: {rmse_lin:.2f}")
    results['linear_reg_r2'] = r2_lin
    results['linear_reg_rmse'] = rmse_lin

    # ---------------------------
    # 2) ЛОГИСТИЧЕСКАЯ РЕГРЕССИЯ
    # ---------------------------
    X_clf, y_clf = prepare_classification_data(df)
    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42
    )
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_clf, y_train_clf)
    y_pred_log = logreg.predict(X_test_clf)
    acc_log = accuracy_score(y_test_clf, y_pred_log)
    f1_log = f1_score(y_test_clf, y_pred_log)
    print(f"  LogisticRegression -> Accuracy: {acc_log:.3f}, F1: {f1_log:.3f}")
    results['logistic_acc'] = acc_log
    results['logistic_f1'] = f1_log

    # ---------------------------
    # 3) ДЕРЕВО РЕШЕНИЙ (регрессия)
    # ---------------------------
    dt_reg = DecisionTreeRegressor(random_state=42, max_depth=5)
    dt_reg.fit(X_train_reg, y_train_reg)
    y_pred_dt = dt_reg.predict(X_test_reg)
    r2_dt = r2_score(y_test_reg, y_pred_dt)
    rmse_dt = mean_squared_error(y_test_reg, y_pred_dt, squared=False)
    print(f"  DecisionTreeRegressor -> R2: {r2_dt:.3f}, RMSE: {rmse_dt:.2f}")
    results['tree_reg_r2'] = r2_dt
    results['tree_reg_rmse'] = rmse_dt

    # ---------------------------
    # 4) CATBOOST (регрессия + классификация)
    # ---------------------------
    # 4.1) CatBoostRegressor
    cbr = CatBoostRegressor(verbose=0, random_state=42)  # verbose=0 чтобы не засорять вывод
    cbr.fit(X_train_reg, y_train_reg)
    y_pred_cbr = cbr.predict(X_test_reg)
    r2_cbr = r2_score(y_test_reg, y_pred_cbr)
    rmse_cbr = mean_squared_error(y_test_reg, y_pred_cbr, squared=False)
    print(f"  CatBoostRegressor -> R2: {r2_cbr:.3f}, RMSE: {rmse_cbr:.2f}")
    results['catboost_reg_r2'] = r2_cbr
    results['catboost_reg_rmse'] = rmse_cbr

    # 4.2) CatBoostClassifier
    cbc = CatBoostClassifier(verbose=0, random_state=42)
    cbc.fit(X_train_clf, y_train_clf)
    y_pred_cbc = cbc.predict(X_test_clf)
    acc_cbc = accuracy_score(y_test_clf, y_pred_cbc)
    f1_cbc = f1_score(y_test_clf, y_pred_cbc)
    print(f"  CatBoostClassifier -> Accuracy: {acc_cbc:.3f}, F1: {f1_cbc:.3f}")
    results['catboost_clf_acc'] = acc_cbc
    results['catboost_clf_f1'] = f1_cbc
    
    return results

# ======================
# ШАГ 5. НЕСКОЛЬКО ВАРИАНТОВ ФИЛЬТРАЦИИ
# ======================
"""
Идея: создадим несколько «версий» DataFrame с разными параметрами фильтрации.
Например, будем менять multiplier в filter_iqr и threshold в filter_z_score.
Проверим, как это влияет на результаты обучения.
"""

filter_versions = {
    # version_name: (iqr_multiplier, z_threshold)
    "ver1_iqr_1.5_z_3": (1.5, 3),
    # можете добавить ещё вариаций
}

results_all = []

for vname, (iqr_mult, z_thr) in filter_versions.items():
    # Копируем изначальный датафрейм
    df_tmp = df_original.copy()

    # 1) Фильтрация IQR
    df_tmp = filter_iqr(df_tmp, 'charges', multiplier=iqr_mult)

    # 2) Фильтрация Z-оценкой
    df_tmp = filter_z_score(df_tmp, 'charges', threshold=z_thr)

    # Можно также фильтровать другие столбцы (например, bmi, age), но осторожно
    # чтобы не урезать датасет слишком сильно.
    
    # Сохраним размер после фильтрации
    print(f"Фильтрация {vname}: осталось {len(df_tmp)} строк из {len(df_original)}")

    # Сохраняем гистограмму «charges» после фильтрации
    plot_histogram(df_tmp, 'charges',
                   f"Распределение `charges` после {vname}",
                   save_path=f"{graphs_dir}/charges_{vname}.png")

    # Пропускаем обучение, если совсем нет данных
    if df_tmp.empty:
        print(f"Версия {vname} дала пустой DataFrame — пропускаем обучение.\n")
        continue

    # Обучаем и собираем результаты
    model_results = train_and_evaluate(df_tmp, version_name=vname)
    model_results["version"] = vname
    model_results["dataset_size"] = len(df_tmp)

    results_all.append(model_results)

# ======================
# ШАГ 6. СВОДИМ РЕЗУЛЬТАТЫ
# ======================
results_df = pd.DataFrame(results_all)
print("\n=== Итоговые результаты по всем версиям фильтрации ===")
print(results_df)

# Сохраняем результаты в CSV, чтобы потом удобно анализировать
results_df.to_csv(f"{graphs_dir}/model_results.csv", index=False)
print(f"\nРезультаты сохранены в {graphs_dir}/model_results.csv")

print("\n=== Рабочий процесс завершён! ===")
