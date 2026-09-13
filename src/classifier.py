from typing import List, Tuple

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import MultinomialNB


class SentimentClassifierNB:
    """
    Классификатор тональности текста на основе Наивного Байесовского метода (Multinomial Naive Bayes).
    """

    def __init__(self):
        self.vectorizer = CountVectorizer()
        self.model = MultinomialNB()
        self.is_trained = False

    def train(self, texts: List[str], labels: List[str]):
        """Обучение векторизатора и Байесовской модели."""
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        self.is_trained = True

    def evaluate(self, texts: List[str], labels: List[str]) -> str:
        """Оценка точности модели."""
        X = self.vectorizer.transform(texts)
        preds = self.model.predict(X)
        acc = accuracy_score(labels, preds)
        report = classification_report(labels, preds, zero_division=0)
        return f"Accuracy: {acc:.2f}\n\nClassification Report:\n{report}"

    def predict(self, text: str) -> Tuple[str, float]:
        """
        Предсказание тональности для одной строки текста.
        Возвращает метку (positive/negative) и уверенность (confidence).
        """
        if not self.is_trained:
            raise RuntimeError("Модель еще не обучена.")

        X = self.vectorizer.transform([text])
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]
        confidence = max(probabilities)
        return prediction, confidence
