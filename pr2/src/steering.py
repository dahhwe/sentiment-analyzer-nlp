from typing import Tuple

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report


class SentimentSteering:
    """
    Стиринг векторов тем с помощью линейного дискриминантного анализа (LDA).
    Темы LSA строятся без учителя и описывают, о чём текст (отели, врачи, автомобили).
    LDA, обученный на метках тональности из ПР1, находит линейные комбинации тем,
    которые лучше всего разделяют классы positive / neutral / negative.
    """

    def __init__(self):
        self.model = LinearDiscriminantAnalysis()
        self.is_trained = False

    def train(self, vectors: np.ndarray, labels):
        self.model.fit(vectors, labels)
        self.is_trained = True

    def evaluate(self, vectors: np.ndarray, labels) -> str:
        preds = self.model.predict(vectors)
        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds, zero_division=0)
        return f"Accuracy: {acc:.2f}\n\nClassification Report:\n{report}"

    def predict(self, vector: np.ndarray) -> Tuple[str, float]:
        """Возвращает: (метка, уверенность)."""
        if not self.is_trained:
            raise RuntimeError("Модель еще не обучена.")
        probs = self.model.predict_proba([vector])[0]
        idx = int(np.argmax(probs))
        return str(self.model.classes_[idx]), float(probs[idx])
