import multiprocessing
from typing import List, Tuple

import numpy as np
from gensim.models import Word2Vec


class WordVectors:
    """
    Векторы слов Word2Vec, обученные на корпусе из ПР1.

    Модель обучается без учителя по предложениям корпуса. В режиме CBOW
    центральное слово окна предсказывается по окружающим его словам,
    в режиме skip-gram, наоборот, окружающие слова предсказываются по центральному.
    Веса скрытого слоя после обучения и есть векторы слов.
    """

    def __init__(self, vector_size: int = 100, window: int = 6, min_count: int = 3,
                 sample: float = 1e-3, sg: int = 0, epochs: int = 20):
        """
        Создает модель с параметрами обучения.

        :param vector_size: размерность векторов слов
        :param window: размер окна контекста
        :param min_count: минимальное число вхождений слова в корпус
        :param sample: порог прореживания часто встречающихся слов
        :param sg: 0 для CBOW, 1 для skip-gram
        :param epochs: число эпох обучения
        """
        self.params = {
            "vector_size": vector_size,
            "window": window,
            "min_count": min_count,
            "sample": sample,
            "sg": sg,
            "epochs": epochs,
        }
        self.model = None

    def train(self, sentences: List[List[str]]):
        """
        Обучает модель Word2Vec.

        :param sentences: список предложений, каждое предложение задано списком лемм
        """
        self.model = Word2Vec(sentences, workers=multiprocessing.cpu_count(), seed=42, **self.params)

    def save(self, path: str):
        """
        Сохраняет обученную модель на диск.

        :param path: путь к файлу модели
        """
        self.model.save(path)

    def load(self, path: str):
        """
        Загружает ранее сохраненную модель.

        :param path: путь к файлу модели
        """
        self.model = Word2Vec.load(path)

    @property
    def vocab_size(self) -> int:
        """
        Размер словаря модели.

        :return: число слов, для которых есть векторы
        """
        return len(self.model.wv)

    def has_word(self, word: str) -> bool:
        """
        Проверяет, есть ли слово в словаре модели.

        :param word: лемма
        :return: True, если для слова есть вектор
        """
        return word in self.model.wv

    def similar_words(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        """
        Находит слова с наиболее близкими векторами.

        :param word: лемма
        :param topn: число слов в ответе
        :return: список пар (слово, косинусное сходство)
        """
        return self.model.wv.most_similar(word, topn=topn)

    def analogy(self, a: str, b: str, c: str, topn: int = 5) -> List[Tuple[str, float]]:
        """
        Решает задачу на аналогию «a относится к b, как ? к c» через вектор a - b + c.

        :param a: первое слово пары-образца
        :param b: второе слово пары-образца
        :param c: слово, для которого ищется пара
        :param topn: число вариантов ответа
        :return: список пар (слово, косинусное сходство)
        """
        return self.model.wv.most_similar(positive=[a, c], negative=[b], topn=topn)

    def odd_one_out(self, words: List[str]) -> str:
        """
        Находит слово, которое по смыслу дальше всего от остальных.

        :param words: список лемм
        :return: лишнее слово
        """
        return self.model.wv.doesnt_match(words)

    def document_vector(self, tokens: List[str]) -> np.ndarray:
        """
        Строит вектор документа как среднее векторов его слов.

        :param tokens: леммы документа
        :return: вектор документа (нулевой, если ни одного слова нет в словаре)
        """
        vectors = [self.model.wv[t] for t in tokens if t in self.model.wv]
        if not vectors:
            return np.zeros(self.model.vector_size)
        return np.mean(vectors, axis=0)
