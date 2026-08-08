# Pre-Entrega N°3 — Clasificador supervisado con TF-IDF (AG News)

## Qué hace este checkpoint
Toma el corpus AG News (mismo dataset y mismo preprocesamiento del Módulo 2), lo vectoriza con
**TF-IDF** y entrena un clasificador clásico para predecir la categoría de cada noticia
(`World`, `Sports`, `Business`, `Sci_Tech`).

## Estructura
```
preentrega3/
├── ag_news/                      # ag_news_train.csv, ag_news_test.csv
├── scripts/
│   ├── preprocessing.py          # regex + lematización SpaCy
│   └── train.py                  # vectorización, comparación de modelos, evaluación
├── figures/
│   └── matriz_confusion.png
├── classification_report.txt
├── comparacion_modelos.csv
├── cargar_dataset.py            
├── requirements.txt
└── README.md
```

## Cómo correrlo
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
cd scripts
python train.py
```

## Decisiones y justificación

### Prevención de data leakage
El `TfidfVectorizer` va **dentro de un `Pipeline`** junto con el clasificador. Durante
`GridSearchCV` (5-fold cross-validation), el `fit_transform` del vectorizador se hace únicamente
sobre los folds de entrenamiento de cada iteración; los datos de test **nunca se usan** para
ajustar el vocabulario ni los pesos IDF. Recién al final, el pipeline ganador —ya entrenado sobre
el 100% de `train`— se usa con `.predict()` (que internamente solo hace `.transform()`, no
`.fit()`) sobre `test`.

### Búsqueda de hiperparámetros del vectorizador
Se probaron combinaciones de:
- `max_features`: `5000`, `10000`, `None` (sin límite)
- `ngram_range`: `(1,1)` (solo unigramas) vs. `(1,2)` (unigramas + bigramas)

En los tres modelos evaluados, la mejor combinación fue **`ngram_range=(1,2)`** — los bigramas
("oil price", "prime minister") aportan señal que los unigramas solos no capturan. El
`max_features=None` (vocabulario completo) ganó en 2 de 3 modelos, lo cual tiene sentido: el
corpus ya viene lematizado y sin stop-words desde el Módulo 2, así que el vocabulario resultante
no es tan ruidoso como para necesitar recortarlo agresivamente.

### Elección del modelo
Se compararon tres clasificadores de referencia, cada uno con su propia búsqueda de
hiperparámetros del vectorizador, usando **F1-macro en cross-validation (5 folds) sobre train**
como criterio de selección (nunca se miró el test hasta elegir el modelo final):

| Modelo | F1-macro (CV, train) | max_features | ngram_range |
|---|---|---|---|
| **LinearSVC** | **0.894** | sin límite | (1,2) |
| MultinomialNB | 0.892 | sin límite | (1,2) |
| LogisticRegression | 0.889 | 10.000 | (1,2) |

Se eligió **LinearSVC** (Support Vector Machine lineal) por tener el mejor F1-macro, aunque la
diferencia con los otros dos es marginal (~0.5 puntos). Es una elección esperable para este tipo
de problema: con vectores TF-IDF de alta dimensión y dispersos, los clasificadores lineales
(SVM lineal, Regresión Logística) suelen superar a Naive Bayes, cuyo supuesto de independencia
condicional entre features es más restrictivo. LinearSVC además es muy eficiente en este régimen
(texto, alta dimensionalidad, pocas muestras relativas a features).

### Resultado final sobre test
**F1-macro = 0.903** (accuracy = 0.903) — por encima del baseline de referencia de la cátedra
(~0.90 con LogisticRegression), confirmando que el preprocesamiento del Módulo 2 (lematización +
remoción de stop-words + fix de entidades HTML) le está dando al modelo tokens de buena calidad.

```
              precision    recall  f1-score   support
    Business      0.847     0.874     0.860       500
    Sci_Tech      0.901     0.872     0.886       500
      Sports      0.951     0.970     0.960       500
       World      0.916     0.898     0.907       500
```

### Análisis de la matriz de confusión
- **Sports es, por lejos, la clase más fácil** (F1 = 0.960): su vocabulario es muy distintivo
  (nombres de equipos, resultados, jugadores) y casi no se confunde con las demás.
- **Business ↔ Sci_Tech es la confusión dominante** (35 Business clasificadas como Sci_Tech, 43
  Sci_Tech clasificadas como Business). Tiene sentido: buena parte de las noticias de tecnología
  en AG News son sobre negocios de empresas tecnológicas (adquisiciones, resultados financieros
  de Microsoft/Google/etc.), por lo que el vocabulario se superpone genuinamente entre ambas
  categorías — no es un error del pipeline sino ambigüedad real del dominio.
- **World se confunde moderadamente con Business** (30 casos): noticias de política internacional
  con impacto económico (comercio, sanciones) comparten vocabulario con ambas categorías.
