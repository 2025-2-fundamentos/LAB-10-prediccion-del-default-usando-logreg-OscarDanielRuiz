# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
#
# Renombre la columna "default payment next month" a "default"
# y remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las demas variables al intervalo [0, 1].
# - Selecciona las K mejores caracteristicas.
# - Ajusta un modelo de regresion logistica.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'type': 'metrics', 'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import os
import json
import gzip
import pickle
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix
)

def read_zip_csv(filepath):
    return pd.read_csv(filepath, compression="zip", index_col=False)

def normalize_education(val):
    return val if val in [1, 2, 3] else 4

def preprocess_frame(df):
    updated = df.rename(columns={"default payment next month": "default"})
    updated = updated[updated["MARRIAGE"] != 0]
    updated = updated[updated["EDUCATION"] != 0]
    updated["EDUCATION"] = updated["EDUCATION"].apply(normalize_education)
    updated.drop(columns=["ID"], inplace=True, errors="ignore")
    return updated

def build_full_pipeline(k_features=23):
    cat = ["SEX", "EDUCATION", "MARRIAGE"]
    encoder = ColumnTransformer(
        [("categorical", OneHotEncoder(handle_unknown="ignore"), cat)],
        remainder="passthrough"
    )

    return Pipeline([
        ("encode", encoder),
        ("scale", MinMaxScaler()),
        ("select", SelectKBest(score_func=f_classif, k=k_features)),
        ("logit", LogisticRegression(max_iter=500, random_state=42)),
    ])

def hyperparam_search(pipe, X, y):
    params = {
        "select__k": range(1, 11),
        "logit__C": [0.001, 0.01, 0.1, 1, 10, 100],
        "logit__penalty": ["l1", "l2"],
        "logit__solver": ["liblinear"],
        "logit__max_iter": [100, 200],
    }
    search = GridSearchCV(
        pipe,
        param_grid=params,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )
    search.fit(X, y)
    return search

def save_compressed_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(model, f)

def make_metrics_dict(y_true, y_pred, label):
    return {
        "type": "metrics",
        "dataset": label,
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "recall": round(recall_score(y_true, y_pred), 4),
        "f1_score": round(f1_score(y_true, y_pred), 4),
    }

def build_confusion_output(y_true, y_pred, label):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "type": "cm_matrix",
        "dataset": label,
        "true_0": {
            "predicted_0": int(cm[0][0]),
            "predicted_1": int(cm[0][1]),
        },
        "true_1": {
            "predicted_0": int(cm[1][0]),
            "predicted_1": int(cm[1][1]),
        },
    }

def export_json_lines(entries, filepath, append=False):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    mode = "a" if append else "w"
    with open(filepath, mode) as f:
        for row in entries:
            f.write(json.dumps(row) + "\n")

df_train = read_zip_csv("files/input/train_data.csv.zip")
df_test = read_zip_csv("files/input/test_data.csv.zip")
df_train = preprocess_frame(df_train)
df_test = preprocess_frame(df_test)

X_train = df_train.drop(columns=["default"])
y_train = df_train["default"]
X_test = df_test.drop(columns=["default"])
y_test = df_test["default"]

model_pipeline = build_full_pipeline(k_features=23)
best_model = hyperparam_search(model_pipeline, X_train, y_train)

save_compressed_model(best_model, "files/models/model.pkl.gz")

train_predictions = best_model.best_estimator_.predict(X_train)
test_predictions = best_model.best_estimator_.predict(X_test)

metrics_list = [
    make_metrics_dict(y_train, train_predictions, "train"),
    make_metrics_dict(y_test, test_predictions, "test"),
]


export_json_lines(metrics_list, "files/output/metrics.json")
cm_train = build_confusion_output(y_train, train_predictions, "train")
cm_test = build_confusion_output(y_test, test_predictions, "test")
export_json_lines([cm_train, cm_test], "files/output/metrics.json", append=True)