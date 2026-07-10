"""Score scaling utilities based on PDO methodology."""

import numpy as np


class ScoreScaler:
    """Convert probabilities/log-odds into scorecard points."""

    def __init__(self, pdo: float = 20.0, base_score: float = 600.0, base_odds: float = 50.0):
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.factor = self.pdo / np.log(2)
        self.offset = self.base_score - self.factor * np.log(self.base_odds)

    def probability_to_score(self, probability: float) -> float:
        probability = np.clip(probability, 1e-6, 1 - 1e-6)
        odds = (1 - probability) / probability
        return float(self.offset + self.factor * np.log(odds))

    def score_to_probability(self, score: float) -> float:
        odds = np.exp((score - self.offset) / self.factor)
        return float(1.0 / (1.0 + odds))
