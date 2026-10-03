from typing import List, Tuple

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

# В ПР1 стоп-слова удаляются до лемматизации, поэтому формы «которые», «всех», «своих»
# проходят фильтр и превращаются в леммы «который», «весь», «свой».
# Для выделения тем такие слова бесполезны, поэтому они убираются уже на этапе TF-IDF.
EXTRA_STOPWORDS = [
    'это', 'этот', 'который', 'весь', 'свой', 'сам', 'самый', 'такой', 'наш', 'ваш', 'мой',
    'мы', 'вы', 'он', 'она', 'они', 'один', 'очень', 'без', 'мочь', 'быть', 'ещё', 'просто',
    'год', 'день', 'время', 'человек', 'сказать',
]


class LSAModel:
    """
    Латентно-семантический анализ (LSA):
    1. Векторы TF-IDF по нормализованным текстам (результат конвейера ПР1).
    2. Усечённое сингулярное разложение (TruncatedSVD) матрицы TF-IDF -> векторы тем документов.
    3. Векторы тем нормализуются по длине (L2), поэтому косинусное сходство
       считается обычным скалярным произведением.
    """

    def __init__(self, n_topics: int = 100):
        self.n_topics = n_topics
        self.vectorizer = None
        self.svd = None
        self.doc_vectors = None
        self.terms = None

    def fit(self, texts: List[str]):
        """Обучение на нормализованных документах (леммы через пробел)."""
        small_corpus = len(texts) < 100  # в маленьком корпусе пользователя редкие слова не отбрасываем
        self.vectorizer = TfidfVectorizer(
            tokenizer=str.split,
            lowercase=False,
            token_pattern=None,
            stop_words=EXTRA_STOPWORDS,
            min_df=1 if small_corpus else 2,      # слово должно встретиться хотя бы в 2 документах
            max_df=1.0 if small_corpus else 0.5,  # и не больше чем в половине документов
            sublinear_tf=True,                    # tf заменяется на 1 + log(tf)
        )
        tfidf = self.vectorizer.fit_transform(texts)
        self.terms = self.vectorizer.get_feature_names_out()

        # Число тем не может быть больше числа документов и слов
        n_topics = max(1, min(self.n_topics, tfidf.shape[0] - 1, tfidf.shape[1] - 1))
        self.svd = TruncatedSVD(n_components=n_topics, n_iter=10, random_state=42)
        self.doc_vectors = normalize(self.svd.fit_transform(tfidf))

    def transform(self, texts: List[str]) -> np.ndarray:
        """Векторы тем для новых текстов."""
        return normalize(self.svd.transform(self.vectorizer.transform(texts)))

    def known_words(self, text: str) -> List[str]:
        """Слова текста, которые есть в словаре модели."""
        return [w for w in text.split() if w in self.vectorizer.vocabulary_]

    def topic_words(self, topic: int, n: int = 8) -> List[str]:
        """Слова с наибольшим весом в теме."""
        weights = self.svd.components_[topic]
        return [self.terms[i] for i in np.argsort(-weights)[:n]]

    def main_topics(self, text: str, n: int = 3) -> List[Tuple[int, float]]:
        """Темы, которые сильнее всего выражены в тексте."""
        vector = self.transform([text])[0]
        best = np.argsort(-vector)[:n]
        return [(int(i), float(vector[i])) for i in best]

    def search(self, text: str, top_n: int = 5) -> List[Tuple[int, float]]:
        """Семантический поиск: косинусное сходство вектора тем запроса с векторами тем документов."""
        query = self.transform([text])[0]
        scores = self.doc_vectors @ query
        best = np.argsort(-scores)[:top_n]
        return [(int(i), float(scores[i])) for i in best if scores[i] > 0]

    def similarity(self, text_a: str, text_b: str) -> Tuple[float, float]:
        """Косинусное сходство двух текстов в пространстве TF-IDF и в пространстве тем LSA."""
        tfidf = self.vectorizer.transform([text_a, text_b])  # строки TF-IDF уже нормированы по L2
        topics = self.transform([text_a, text_b])
        return float(tfidf[0].multiply(tfidf[1]).sum()), float(topics[0] @ topics[1])
