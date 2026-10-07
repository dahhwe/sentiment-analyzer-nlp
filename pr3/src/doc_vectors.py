import multiprocessing
from typing import List

import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.preprocessing import normalize


class DocVectors:
    """
    Векторы документов Doc2Vec.

    Каждому документу корпуса сопоставляется собственный вектор, который обучается
    вместе с моделью. Используется вариант PV-DBOW (dm=0): по вектору документа
    предсказываются слова этого документа. Вектор нового документа выводится
    методом infer_vector при «замороженных» весах модели.
    """

    def __init__(self, vector_size: int = 100, min_count: int = 2, epochs: int = 40, dm: int = 0):
        """
        Создает модель с параметрами обучения.

        :param vector_size: размерность векторов документов
        :param min_count: минимальное число вхождений слова в корпус
        :param epochs: число эпох обучения
        :param dm: 0 для PV-DBOW, 1 для PV-DM
        """
        self.params = {
            "vector_size": vector_size,
            "min_count": min_count,
            "epochs": epochs,
            "dm": dm,
        }
        self.model = None

    def train(self, documents: List[List[str]]):
        """
        Обучает модель Doc2Vec. Тегом документа служит его номер в корпусе.

        :param documents: список документов, каждый документ задан списком лемм
        """
        tagged = [TaggedDocument(words, [i]) for i, words in enumerate(documents)]
        self.model = Doc2Vec(workers=multiprocessing.cpu_count(), seed=42, **self.params)
        self.model.build_vocab(tagged)
        self.model.train(tagged, total_examples=self.model.corpus_count, epochs=self.model.epochs)

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
        self.model = Doc2Vec.load(path)

    @property
    def corpus_size(self) -> int:
        """
        Число документов, на которых обучалась модель.

        :return: количество векторов документов
        """
        return len(self.model.dv)

    def corpus_vectors(self) -> np.ndarray:
        """
        Векторы документов обучающего корпуса, нормированные по длине.

        :return: матрица размера (число документов, размерность)
        """
        return normalize(np.array([self.model.dv[i] for i in range(self.corpus_size)]))

    def infer(self, tokens: List[str]) -> np.ndarray:
        """
        Выводит вектор нового документа.

        :param tokens: леммы документа
        :return: вектор документа
        """
        return self.model.infer_vector(tokens)
