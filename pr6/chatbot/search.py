import copy
from operator import itemgetter
from typing import List, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from voyager import Index, Space


def fine_tune_word_vectors(word_vectors, sentences: List[List[str]], epochs: int = 20):
    """
    Дообучает копию модели Word2Vec из ПР3 на текстах предметной области.

    Корпус отзывов ПР1 не содержит слов «рендер», «разметка», «Blender», «YOLO», а «огонь»
    там всегда заменялся на «хороший», поэтому около четверти лемм диалогов не имели векторов.
    Новые слова добавляются в словарь, и модель проходит по текстам README и корпуса диалогов
    еще несколько эпох. Сама модель ПР3 не меняется, ее по-прежнему используют сети тональности.

    :param word_vectors: модель WordVectors из ПР3
    :param sentences: предложения предметной области в виде списков лемм
    :param epochs: число эпох дообучения
    :return: новая модель WordVectors
    """
    tuned = copy.deepcopy(word_vectors)
    tuned.model.min_count = 1
    tuned.model.build_vocab(sentences, update=True)
    tuned.model.train(sentences, total_examples=len(sentences), epochs=epochs)
    return tuned


class HybridSearch:
    """
    Поиск похожих текстов по двум представлениям сразу.

    1. TF-IDF, как в чат-боте на основе информационного поиска из лекции 12. Векторы
       разреженные, поэтому сходство со всеми документами считается быстро и точно.
    2. Среднее векторов Word2Vec: находит документы с близкими по смыслу словами, даже если
       общих слов нет. Векторы плотные, и для них строится индекс приближенного поиска
       ближайших соседей Voyager (HNSW) из лекции 13.

    Кандидаты из обоих поисков объединяются, итоговая оценка равна
    alpha * сходство_TF-IDF + (1 - alpha) * сходство_Word2Vec.
    """

    def __init__(self, word_vectors, alpha: float = 0.7, candidates: int = 20):
        """
        Создает поиск без проиндексированных документов.

        :param word_vectors: модель WordVectors (дообученная на текстах предметной области)
        :param alpha: вес сходства TF-IDF в итоговой оценке
        :param candidates: сколько кандидатов брать из каждого поиска
        """
        self.word_vectors = word_vectors
        self.alpha = alpha
        self.candidates = candidates
        self.vectorizer = None
        self.tfidf = None
        self.dense = None
        self.index = None

    def dense_vectors(self, documents: List[List[str]]) -> np.ndarray:
        """
        Средние векторы Word2Vec документов, нормированные по длине.

        :param documents: документы в виде списков лемм
        :return: массив (число документов, размерность векторов)
        """
        vectors = np.array([self.word_vectors.document_vector(doc) for doc in documents], dtype="float32")
        return normalize(vectors + 1e-9).astype("float32")

    def fit(self, documents: List[List[str]]):
        """
        Индексирует документы. Векторизатор TF-IDF настроен как в ПР2, но без порога min_df=2:
        он выбрасывает слова, которые встречаются только в одном высказывании, а в коротких
        репликах это часто самые характерные слова.

        :param documents: документы в виде списков лемм
        """
        self.vectorizer = TfidfVectorizer(tokenizer=str.split, lowercase=False, token_pattern=None, sublinear_tf=True)
        self.tfidf = self.vectorizer.fit_transform([" ".join(doc) for doc in documents])
        self.dense = self.dense_vectors(documents)
        self.index = Index(Space.Cosine, num_dimensions=self.dense.shape[1], random_seed=1337)
        self.index.add_items(self.dense)

    def exact_scores(self, query: List[str]) -> np.ndarray:
        """
        Итоговые оценки для всех документов полным перебором (для проверки индекса).

        :param query: леммы запроса
        :return: массив оценок по всем документам
        """
        tfidf = (self.vectorizer.transform([" ".join(query)]) @ self.tfidf.T).toarray()[0]
        return self.alpha * tfidf + (1 - self.alpha) * (self.dense @ self.dense_vectors([query])[0])

    def search(self, query: List[str], k: int = 5, use_index: bool = True) -> List[Tuple[int, float]]:
        """
        Находит k самых похожих документов.

        :param query: леммы запроса
        :param k: число результатов
        :param use_index: брать кандидатов Word2Vec из индекса Voyager (иначе полный перебор)
        :return: список пар (номер документа, оценка) по убыванию оценки
        """
        if not use_index:
            scores = self.exact_scores(query)
            best = np.argsort(-scores, kind="stable")[:k]
            return [(int(i), float(scores[i])) for i in best]
        tfidf = (self.vectorizer.transform([" ".join(query)]) @ self.tfidf.T).toarray()[0]
        dense_query = self.dense_vectors([query])[0]
        n = min(self.candidates, len(self.dense))
        ids, _ = self.index.query(dense_query, k=n)
        candidates = set(int(i) for i in ids) | set(int(i) for i in np.argsort(-tfidf)[:n])
        scored = [(i, self.alpha * tfidf[i] + (1 - self.alpha) * float(self.dense[i] @ dense_query)) for i in candidates]
        return sorted(scored, key=itemgetter(1), reverse=True)[:k]
