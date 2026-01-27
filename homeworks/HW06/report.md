# HW06 – Report

> Файл: `homeworks/HW06/report.md`  


## 1. Dataset

- Какой датасет выбран: `S06-hw-dataset-04.csv`
- Размер: (25000, 62)
- Целевая переменная: `target` (Class 0: 95.08%, Class 1: 4.92%)
- Признаки: Все числовые признаки, синтетические данные с сильным дисбалансом классов.

## 2. Protocol

- Разбиение: 80% train / 20% test, `random_state=42`, `stratify=y`
- Подбор: 5-fold StratifiedKFold на train, оптимизация по F1-score (критично для дисбаланса)
- Метрики: accuracy (базовая метрика), F1 (учитывает дисбаланс классов), ROC-AUC (показывает качество ранжирования вероятностей). Выбор обусловлен необходимостью оценки качества при сильном дисбалансе классов.

## 3. Models

Сравнивались следующие модели:

1. DummyClassifier (baseline, strategy='stratified')
2. LogisticRegression (через pipeline со StandardScaler, class_weight='balanced')
3. DecisionTreeClassifier (подбор max_depth [3,5,7,10] и min_samples_leaf [5,10,20])
4. RandomForestClassifier (подбор max_depth [10,15,None], min_samples_leaf [2,5], max_features ['sqrt','log2'])
5. HistGradientBoostingClassifier (современная реализация градиентного бустинга)
6. StackingClassifier (DecisionTree + RandomForest + HistGradientBoosting в качестве базовых моделей, LogisticRegression с class_weight='balanced' как метамодель)

Все модели обучались с параметром class_weight='balanced' для компенсации дисбаланса классов. Наилучшие гиперпараметры:

- DecisionTree: max_depth=10, min_samples_leaf=5
- RandomForest: max_depth=10, max_features='sqrt', min_samples_leaf=5

## 4. Results

**Финальные метрики на тестовой выборке:**
| Модель | Accuracy | F1 | ROC-AUC |
|--------|----------|-----|---------|
| DummyClassifier | 0.9050 | 0.0365 | 0.4933 |
| LogisticRegression | 0.7756 | 0.2540 | 0.8355 |
| DecisionTree | 0.8948 | 0.3826 | 0.7860 |
| RandomForest | 0.9746 | **0.7171** | 0.9014 |
| HistGradientBoosting | **0.9792** | **0.7374** | 0.8865 |
| StackingClassifier | 0.9174 | 0.4920 | **0.9023** |

Победитель: **HistGradientBoosting** по метрике ROC-AUC на кросс-валидации (0.9002), однако на тестовой выборке **StackingClassifier** показал лучший результат по ROC-AUC (0.9023). HistGradientBoosting выигрывает по F1 (0.7374) и accuracy (0.9792), что указывает на его превосходство в классификации majority-класса.

## 5. Analysis

**Устойчивость:** Анализ с различными random_state (10, 42, 100, 200, 300) показал, что разброс ROC-AUC для HistGradientBoosting составил ±0.012, что говорит о стабильности решения. StackingClassifier демонстрировал больший разброс по ROC-AUC (±0.018).

**Ошибки:** Confusion matrix для StackingClassifier:
```
[[4750   4]
 [ 100 146]]
```
Матрица показывает: 4750 True Negative, 4 False Positive, 100 False Negative, 146 True Positive. Высокая точность (97.92%), но модель имеет относительно большое количество ложноотрицательных результатов (100 FN), что отражает сложность выявления редкого класса. Это типично для задач с дисбалансом классов и объясняет, почему F1-score (0.7374) ниже, чем можно было бы ожидать при такой высокой accuracy.

**Интерпретация:** Permutation importance выявил следующие top-10 признаков:
1. f58
2. f25
3. f54
4. f47
5. f38
6. f53
7. f13
8. f04
9. f11
10. f33

Признаки f58, f25 и f54 имеют наибольшее влияние на предсказания модели. Это согласуется с синтетической природой данных и подтверждает, что модель фокусируется на аномальных паттернах для выявления редкого класса.

## 6. Conclusion

1. При сильном дисбалансе классов accuracy является вводящей в заблуждение метрикой: RandomForest показывает 97.46% accuracy, но это в основном за счет правильного предсказания majority-класса.
   
2. Компромиссы между метриками критичны для выбора модели: HistGradientBoosting выигрывает по F1 и accuracy, а StackingClassifier - по ROC-AUC (качество ранжирования).

3. Ансамбли (RandomForest, HistGradientBoosting, Stacking) существенно превосходят одиночные модели (DecisionTree, LogisticRegression), что демонстрирует их эффективность при работе с зашумленными и несбалансированными данными.

4. Честный ML-протокол с фиксированным разделением данных, кросс-валидацией на тренировочной выборке и однократной оценкой на тестовой выборке необходим для получения объективных результатов.

5. Использование class_weight='balanced' во всех моделях критически важно для дисбалансированных данных, без этого большинство моделей игнорировали бы minority-класс.

6. HistGradientBoosting показал лучшее качество классификации (accuracy и F1), но StackingClassifier обеспечил лучшее ранжирование вероятностей (ROC-AUC), что подчеркивает важность выбора критерия оценки в зависимости от бизнес-задачи.