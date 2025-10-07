import pandas as pd
from typing import Tuple, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_validate, GridSearchCV
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def random_forest(X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """

    Aplica o SMOTE para equilibrar a distribuição da classe minoritária, os rebaixados, dentro de uma pipeline
    para não causar data leak,identifica os parâmetros mais adequados para o modelo de Random Forest através de
    um hyperparameter tuning, treina e avalia o modelo de Random Forest.
    """

    # Define o Pipeline
    pipeline = Pipeline([('smote', SMOTE(random_state=42)), ('modelo', RandomForestClassifier(random_state=42))])

    # Define os parâmetros que serão testados
    param_grid = {'modelo__n_estimators': [100, 200, 500],
                  'modelo__max_depth': [None, 5, 10, 20],
                  'modelo__min_samples_split': [2, 5, 10],
                  'modelo__min_samples_leaf': [1, 2, 4],
                  'modelo__max_features': ['sqrt', 'log2'],
                  'modelo__class_weight': ['balanced']
                  }

    # Define quais são os melhores parâmetros
    grid = GridSearchCV(estimator=pipeline, param_grid=param_grid, cv=5, scoring='f1_macro', n_jobs=-1)

    grid.fit(X, y)

    melhores_parametros = grid.best_params_

    melhor_modelo = grid.best_estimator_

    # Define quais as métricas que serão visualizadas, focadas nos times rebaixados
    avaliacao = {'precisao_rebaixado': make_scorer(precision_score, pos_label=0),
                 'recal_rebaixado': make_scorer(recall_score, pos_label=0),
                 'f1_rebaixado': make_scorer(f1_score, pos_label=0), 'acuracia': 'accuracy'}

    # Treina o modelo
    resultados = cross_validate(melhor_modelo, X, y, cv=5, scoring=avaliacao, return_train_score=False)

    # Calcula as médias das validações
    metricas = {m: resultados['test_' + m].mean() for m in avaliacao.keys()}

    print("\nMétricas Médias (Random Forest):")
    for nome, valor in metricas.items():
        print(f"{nome}: {valor:.2f}")

    print("\nMelhores parâmetros encontrados (Random Forest):")
    for nome, valor in melhores_parametros.items():
        print(f"{nome}: {valor}")

    return metricas, melhores_parametros

def gradient_boosting(X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """

    Aplica o SMOTE para equilibrar a distribuição da classe minoritária, os rebaixados, dentro de uma pipeline
    para não causar data leak,identifica os parâmetros mais adequados para o modelo de Gradient Boosting,
    através de um hyperparameter tuning, treina e avalia o modelo de Gradient Boosting.
    """

    # Define o Pipeline
    pipeline = Pipeline([('smote', SMOTE(random_state=42)), ('modelo', GradientBoostingClassifier(random_state=42))])

    # Define os parâmetros que serão testados
    param_grid = {
        'modelo__n_estimators': [10, 50, 100, 500],
        'modelo__learning_rate': [0.0001, 0.001, 0.01, 0.1, 1.0],
        'modelo__max_depth': [3, 5, 7, 9]
    }

    # Define quais são os melhores parâmetros
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='f1_macro'
    )
    grid.fit(X, y)

    melhor_modelo = grid.best_estimator_

    melhores_parametros = grid.best_params_

    # Define quais as métricas que serão visualizadas, focadas nos times rebaixados
    avaliacao = {
        'precisao_rebaixado': make_scorer(precision_score, pos_label=0),
        'recal_rebaixado': make_scorer(recall_score, pos_label=0),
        'f1_rebaixado': make_scorer(f1_score, pos_label=0),
        'acuracia': 'accuracy'
    }

    # Treina o modelo
    resultados = cross_validate(
        melhor_modelo, X, y, cv=5, scoring=avaliacao, return_train_score=False
    )

    # Calcula as médias das validações
    metricas = {m: resultados['test_' + m].mean() for m in avaliacao.keys()}

    print("\nMétricas Médias (Gradient Boosting):")
    for nome, valor in metricas.items():
        print(f"{nome}: {valor:.2f}")

    print("\nMelhores parâmetros encontrados (Gradient Boosting):")
    for nome, valor in melhores_parametros.items():
        print(f"{nome}: {valor}")

    return metricas, melhores_parametros

def logistic_regression(X: pd.DataFrame, y: pd.Series) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """

    Faz um escalonamento dos dados, aplica o SMOTE para equilibrar a distribuição da classe minoritária,
    os rebaixados, dentro de uma pipeline para não causar data leak,identifica os parâmetros mais
    adequados para o modelo de Regressão Logística, através de um hyperparameter tuning, treina e
    avalia o modelo de Regressão Logística.
    """

    # Define o Pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('modelo', LogisticRegression(class_weight='balanced', random_state=42, max_iter=500))
    ])

    # Define os parâmetros que serão testados
    param_grid = {
        'modelo__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'modelo__solver': ['liblinear', 'lbfgs', 'saga'],
        'modelo__penalty': ['l2']
    }

    # Define quais são os melhores parâmetros
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring='f1_macro'
    )

    grid.fit(X, y)

    melhores_parametros = grid.best_params_

    melhor_modelo = grid.best_estimator_

    # Define quais as métricas que serão visualizadas, focadas nos times rebaixados
    avaliacao = {
        'precisao_rebaixado': make_scorer(precision_score, pos_label=0),
        'recal_rebaixado': make_scorer(recall_score, pos_label=0),
        'f1_rebaixado': make_scorer(f1_score, pos_label=0),
        'acuracia': 'accuracy'
    }

    # Treina o modelo
    resultados = cross_validate(
        melhor_modelo, X, y, cv=5, scoring=avaliacao, return_train_score=False
    )

    # Calcula as médias das validações
    metricas = {m: resultados['test_' + m].mean() for m in avaliacao.keys()}

    print("\nMétricas Médias (Regressão Logística):")
    for nome, valor in metricas.items():
        print(f"{nome}: {valor:.2f}")

    print("\nMelhores parâmetros encontrados (Regressão Logística):")
    for nome, valor in melhores_parametros.items():
        print(f"{nome}: {valor}")

    return metricas, melhores_parametros


