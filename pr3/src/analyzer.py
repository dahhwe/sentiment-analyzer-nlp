from typing import Dict, List, Tuple

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

METHOD_NAMES = {
    "lsa": "LSA (ПР2)",
    "word2vec": "Word2Vec (среднее векторов слов)",
    "doc2vec": "Doc2Vec (PV-DBOW)",
}


class DocumentAnalyzer:
    """
    Семантический анализатор набора документов.

    Объединяет анализатор из ПР2 (векторы тем LSA) и векторы Word2Vec и Doc2Vec из ПР3,
    чтобы одни и те же операции (поиск, сравнение, обработка набора документов)
    можно было выполнить каждым способом и сравнить результаты.
    """

    def __init__(self, lsa, word_vectors, doc_vectors, corpus_docs: List[List[str]], tokenizer, normalizer):
        """
        Создает анализатор из уже обученных моделей.

        :param lsa: обученная модель LSAModel из ПР2
        :param word_vectors: обученная модель WordVectors
        :param doc_vectors: обученная модель DocVectors
        :param corpus_docs: документы корпуса в виде списков лемм
        :param tokenizer: токенизатор RegexTokenizer из ПР1
        :param normalizer: нормализатор TextNormalizer из ПР1
        """
        self.lsa = lsa
        self.word_vectors = word_vectors
        self.doc_vectors = doc_vectors
        self.corpus_docs = corpus_docs
        self.tokenizer = tokenizer
        self.normalizer = normalizer
        self._corpus_vectors = {}

    def normalize_text(self, text: str) -> List[str]:
        """
        Переводит текст в список лемм конвейером ПР1.

        :param text: исходный текст
        :return: список лемм
        """
        return self.normalizer.normalize(self.tokenizer.tokenize(text))

    def vectors(self, documents: List[List[str]], method: str) -> np.ndarray:
        """
        Строит векторы для набора документов выбранным способом.

        :param documents: документы в виде списков лемм
        :param method: "lsa", "word2vec" или "doc2vec"
        :return: матрица векторов, нормированных по длине
        """
        if method == "lsa":
            return self.lsa.transform([" ".join(doc) for doc in documents])
        if method == "word2vec":
            return normalize(np.array([self.word_vectors.document_vector(doc) for doc in documents]))
        return normalize(np.array([self.doc_vectors.infer(doc) for doc in documents]))

    def corpus_vectors(self, method: str) -> np.ndarray:
        """
        Векторы всех документов корпуса (вычисляются один раз и запоминаются).

        :param method: "lsa", "word2vec" или "doc2vec"
        :return: матрица векторов, нормированных по длине
        """
        if method not in self._corpus_vectors:
            if method == "lsa":
                self._corpus_vectors[method] = self.lsa.doc_vectors
            elif method == "doc2vec":
                self._corpus_vectors[method] = self.doc_vectors.corpus_vectors()
            else:
                self._corpus_vectors[method] = self.vectors(self.corpus_docs, method)
        return self._corpus_vectors[method]

    def search(self, query: str, method: str, top_n: int = 3) -> List[Tuple[int, float]]:
        """
        Ищет в корпусе документы, близкие по смыслу к запросу.

        :param query: текст запроса
        :param method: "lsa", "word2vec" или "doc2vec"
        :param top_n: число найденных документов
        :return: список пар (номер документа, косинусное сходство)
        """
        query_vector = self.vectors([self.normalize_text(query)], method)[0]
        scores = self.corpus_vectors(method) @ query_vector
        best = np.argsort(-scores)[:top_n]
        return [(int(i), float(scores[i])) for i in best]

    def compare(self, text_a: str, text_b: str) -> Dict[str, float]:
        """
        Сравнивает два текста всеми способами.

        :param text_a: первый текст
        :param text_b: второй текст
        :return: словарь {способ: косинусное сходство}
        """
        docs = [self.normalize_text(text_a), self.normalize_text(text_b)]
        result = {}
        for method in METHOD_NAMES:
            vectors = self.vectors(docs, method)
            result[method] = float(vectors[0] @ vectors[1])
        return result

    def lsa_similar_words(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Находит близкие слова по векторам тем слов LSA (строки матрицы V*S).

        Кандидатами считаются только слова, которые есть и в словаре Word2Vec,
        чтобы оба анализатора сравнивались на одном наборе слов.

        :param word: лемма
        :param topn: число слов в ответе
        :return: список пар (слово, косинусное сходство) или пустой список
        """
        terms = list(self.lsa.terms)
        if word not in terms:
            return []
        word_matrix = normalize(self.lsa.svd.components_.T * self.lsa.svd.singular_values_)
        scores = word_matrix @ word_matrix[terms.index(word)]
        result = []
        for i in np.argsort(-scores):
            if terms[i] != word and self.word_vectors.has_word(terms[i]):
                result.append((terms[i], float(scores[i])))
            if len(result) == topn:
                break
        return result

    def process_documents(self, documents: List[str], method: str, n_clusters: int):
        """
        Обрабатывает набор документов: матрица попарного сходства и разбиение на группы.

        Группы находятся методом k-средних по векторам документов.

        :param documents: тексты набора
        :param method: "lsa", "word2vec" или "doc2vec"
        :param n_clusters: число групп
        :return: (матрица косинусного сходства, номер группы для каждого документа)
        """
        vectors = self.vectors([self.normalize_text(doc) for doc in documents], method)
        matrix = vectors @ vectors.T
        n_clusters = max(1, min(n_clusters, len(documents)))
        clusters = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(vectors)
        return matrix, clusters
