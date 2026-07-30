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
results = pipeline.evaluate(X, y)
```

## Preprocesamiento según modelo

El pipeline aplica un preprocesamiento diferente según el tipo de modelo seleccionado:

| `model_type` | Preprocesamiento |
|---|---|
| `logistic` | Encoding → Binning → WOE |
| `neural_network` | Encoding → StandardScaler |
| `random_forest` | Encoding (sin transformación adicional) |
| `xgboost` | Encoding (sin transformación adicional) |

- **Regresión logística:** es el único modelo que usa binning y transformación WOE. Esto permite generar variables interpretables y es el enfoque clásico de scorecards.
- **Red neuronal (MLP):** se aplica `StandardScaler` para estandarizar las variables numéricas, lo cual mejora la convergencia del entrenamiento.
- **Random Forest y XGBoost:** no necesitan binning ni estandarización; se entrenan directamente con las variables numéricas tal como vienen (después de encoding de categóricas e imputación de missings).

## Configurar el pipeline sin usar el config por defecto

`CreditScoringPipeline` puede inicializarse de 3 maneras:

1. **Sin parámetros** (usa el config por defecto):
   ```python
   pipeline = CreditScoringPipeline()
   ```
2. **Con `PipelineConfig`** (configuración tipada):
   ```python
   from creditScoring import CreditScoringPipeline, PipelineConfig

   custom = PipelineConfig(model_type="random_forest", random_state=7)
   pipeline = CreditScoringPipeline(config=custom)
   ```
3. **Con `dict`** (control total de secciones y valores):
   ```python
   from creditScoring import CreditScoringPipeline

   custom = {
       "binning": {"n_bins": 10, "method": "equal_width"},
       "missing": {
           "numeric_strategy": "mean",
           "categorical_strategy": "constant",
           "fill_value": -1,
           "categorical_fill_value": "NA",
       },
       "feature_selection": {
           "min_iv": 0.03,
           "max_correlation": 0.75,
           "correlation_method": "spearman",
       },
       "evaluation": {"threshold": 0.45},
       "model_type": "logistic",
       "random_state": 123,
   }
   pipeline = CreditScoringPipeline(config=custom)
   ```

### Opciones disponibles más usadas

- `model_type`: `logistic`, `random_forest`, `neural_network`, `xgboost`
- `binning.method`: `quantile`, `equal_width`, `supervised`, `monotonic` (solo aplica para `logistic`)
- `missing.numeric_strategy`: `mean`, `median`, `replace`, `drop`
- `missing.categorical_strategy`: `mode`, `constant`, `drop`

Si usas `binning.method` con `supervised` o `monotonic`, el pipeline requiere `y` durante el preprocesamiento de entrenamiento.

## Estructura de módulos

- `config`: configuración por defecto y configuración tipada.
- `data`: carga y validación de dataframes.
- `preprocessing`: missing values, encoding, binning y WoE.
- `feature_selection`: selección por IV y filtro de correlación.
- `models`: wrappers para Logistic Regression, Random Forest, XGBoost y MLP.
- `evaluation`: métricas de clasificación y crédito.
- `pipeline`: flujo end-to-end de entrenamiento y scoring.
- `visualization`: funciones de gráficas para análisis y performance.
