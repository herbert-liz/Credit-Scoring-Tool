# Credit-Scoring-Tool

Paquetería en Python para estandarizar y facilitar el desarrollo de modelos de score crediticio.

## Instalación

```bash
pip install credit-scoring-tool
```

Para desarrollo local:

```bash
pip install -e .
```

## Quickstart

```python
import pandas as pd
from creditScoring.pipeline import CreditScoringPipeline

X = pd.DataFrame({
    "income": [1000, 1200, 900, 2500],
    "age": [25, 32, 45, 29],
    "segment": ["A", "B", "B", "A"],
})
y = pd.Series([1, 0, 0, 1])

pipeline = CreditScoringPipeline()
pipeline.fit(X, y)
prob = pipeline.predict_proba(X)
score = pipeline.score(X)
results = pipeline.evaluate(X, y)
```

## Estructura de módulos

- `config`: configuración por defecto y configuración tipada.
- `data`: carga y validación de dataframes.
- `preprocessing`: missing values, encoding, binning y WoE.
- `feature_selection`: selección por IV y filtro de correlación.
- `models`: wrappers para Logistic Regression, Random Forest, XGBoost y MLP.
- `evaluation`: métricas de clasificación y crédito.
- `scorecard`: escalamiento PDO y generación de puntos.
- `pipeline`: flujo end-to-end de entrenamiento y scoring.
- `visualization`: funciones de gráficas para análisis y performance.
