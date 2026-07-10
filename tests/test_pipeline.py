import pandas as pd

from creditScoring.pipeline import CreditScoringPipeline


def test_pipeline_fit_predict_evaluate():
    X = pd.DataFrame(
        {
            "income": [1000, 1200, 900, 2500, 1900, 2100, 800, 1300],
            "age": [25, 32, 45, 29, 37, 41, 22, 30],
            "segment": ["A", "B", "B", "A", "C", "A", "B", "C"],
        }
    )
    y = pd.Series([1, 0, 0, 1, 0, 1, 0, 1])

    pipe = CreditScoringPipeline()
    pipe.fit(X, y)

    proba = pipe.predict_proba(X)
    scores = pipe.score(X)
    results = pipe.evaluate(X, y)

    assert len(proba) == len(X)
    assert len(scores) == len(X)
    assert "roc_auc" in results.classification_metrics
