"""
train.py — Pre-Entrega N°3: Clasificador supervisado con TF-IDF (AG News)
"""

import sys, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, f1_score

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from cargar_dataset import cargar
from preprocessing import preprocess_corpus

RANDOM_STATE = 42

# 1. Carga de datos (splits provistos por la cátedra, no se tocan entre sí)
train_df, test_df = cargar()

# 2. Preprocesamiento (mismas funciones del Módulo 2: regex + lematización SpaCy)
train_df["text_clean"] = preprocess_corpus(train_df["text"].tolist())
test_df["text_clean"] = preprocess_corpus(test_df["text"].tolist())

X_train, y_train = train_df["text_clean"], train_df["label"]
X_test, y_test = test_df["text_clean"], test_df["label"]

# 3. Búsqueda de la mejor combinación (vectorizador + modelo) SOLO con datos de train.
#    GridSearchCV hace fit_transform del TF-IDF únicamente sobre los folds de train
#    en cada iteración de la cross-validation -> el test nunca se toca acá.
param_grid_common = {
    "tfidf__max_features": [5000, 10000, None],
    "tfidf__ngram_range": [(1, 1), (1, 2)],
}

candidatos = {
    "LogisticRegression": (LogisticRegression(max_iter=1000, random_state=RANDOM_STATE), {}),
    "MultinomialNB": (MultinomialNB(), {}),
    "LinearSVC": (LinearSVC(random_state=RANDOM_STATE), {}),
}

resultados = []
mejores_estimadores = {}

for nombre, (clf, extra_grid) in candidatos.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", clf),
    ])
    grid = {**param_grid_common, **extra_grid}
    gs = GridSearchCV(pipe, grid, scoring="f1_macro", cv=5, n_jobs=-1)
    gs.fit(X_train, y_train)

    resultados.append({
        "modelo": nombre,
        "f1_macro_cv": gs.best_score_,
        "max_features": gs.best_params_["tfidf__max_features"],
        "ngram_range": gs.best_params_["tfidf__ngram_range"],
    })
    mejores_estimadores[nombre] = gs.best_estimator_
    print(f"{nombre}: f1_macro(cv)={gs.best_score_:.4f} | params={gs.best_params_}")

resultados_df = pd.DataFrame(resultados).sort_values("f1_macro_cv", ascending=False)
resultados_df.to_csv("comparacion_modelos.csv", index=False)
print("\nComparación de modelos (ordenado por f1_macro en cross-validation):")
print(resultados_df.to_string(index=False))

# 4. Modelo ganador: se evalúa UNA sola vez sobre el test, ya vectorizado con
#    el TF-IDF ajustado (fit) exclusivamente sobre train (dentro del Pipeline).
mejor_nombre = resultados_df.iloc[0]["modelo"]
mejor_pipeline = mejores_estimadores[mejor_nombre]
print(f"\nModelo elegido: {mejor_nombre}")

y_pred = mejor_pipeline.predict(X_test)

reporte = classification_report(y_test, y_pred, digits=3)
reporte_dict = classification_report(y_test, y_pred, digits=3, output_dict=True)
print("\nClassification report (test):")
print(reporte)

with open("classification_report.txt", "w") as f:
    f.write(f"Modelo: {mejor_nombre}\n")
    f.write(f"Vectorizador: {mejor_pipeline.named_steps['tfidf']}\n\n")
    f.write(reporte)

with open("classification_report.json", "w") as f:
    json.dump(reporte_dict, f, indent=2)

# 5. Matriz de confusión
clases = sorted(y_test.unique())
cm = confusion_matrix(y_test, y_pred, labels=clases)

fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(clases))); ax.set_xticklabels(clases, rotation=45, ha="right")
ax.set_yticks(range(len(clases))); ax.set_yticklabels(clases)
ax.set_xlabel("Predicción"); ax.set_ylabel("Real")
ax.set_title(f"Matriz de confusión — {mejor_nombre} (test)")
for i in range(len(clases)):
    for j in range(len(clases)):
        color = "white" if cm[i, j] > cm.max() / 2 else "black"
        ax.text(j, i, cm[i, j], ha="center", va="center", color=color)
fig.colorbar(im)
plt.tight_layout()
plt.savefig("figures/matriz_confusion.png")
plt.close()

print(f"\nF1-macro final (test): {f1_score(y_test, y_pred, average='macro'):.4f}")
print("Listo: classification_report.txt, comparacion_modelos.csv, figures/matriz_confusion.png")
