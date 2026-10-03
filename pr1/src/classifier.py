from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB


class SentimentClassifierNB:
    """
    Классификатор тональности текста на основе Наивного Байеса.
    Поддерживает 3 класса: positive, negative, neutral.
    Устойчив к OOV (словам вне словаря) и пограничным неопределенным случаям.
    """

    def __init__(self, uncertainty_threshold: float = 0.15):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.is_trained = False
        self.uncertainty_threshold = uncertainty_threshold

    def train(self, texts: List[str], labels: List[str]):
        """Обучение векторизатора и модели."""
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True

    def evaluate(self, texts: List[str], labels: List[str]) -> str:
        """Оценка точности классификатора."""
        X = self.vectorizer.transform(texts)
        preds = self.model.predict(X)
        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds, zero_division=0)
        return f"Accuracy: {acc:.2f}\n\nClassification Report:\n{report}"

    def predict(self, text: str) -> Tuple[str, float, str]:
        """
        Предсказание тональности текста.
        Возвращает: (метка, уверенность, комментарий)
        """
        if not self.is_trained:
            raise RuntimeError("Модель еще не обучена.")

        X = self.vectorizer.transform([text])

        if X.nnz == 0:
            return "neutral", 1.0, "Эмоциональные маркеры не обнаружены (фактический/нейтральный текст)"

        probs = self.model.predict_proba(X)[0]
        classes = self.model.classes_
        max_idx = int(np.argmax(probs))
        pred_label = classes[max_idx]
        confidence = float(probs[max_idx])

        sorted_probs = np.sort(probs)
        margin = sorted_probs[-1] - sorted_probs[-2]
        if margin < self.uncertainty_threshold and pred_label != "neutral":
            return "neutral", confidence, "Текст содержит противоречивые признаки (классифицирован как нейтральный)"

        return pred_label, confidence, "Уверенная классификация"
